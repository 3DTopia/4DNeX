
import torch
import torch.nn.functional as F

import glob
import pickle
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import imageio
from core.dataset import PointmapDataset
from core.annotation import Monst3RAnno
import tyro

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Batched Zhou-2019 6-D → SO(3).
      d6 : (B,6)  →  R : (B,3,3)
    """
    a1, a2 = d6[:, 0:3], d6[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - (a2 * b1).sum(-1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)          # (B,3,3)


def build_K(fx, fy, cx, cy) -> torch.Tensor:
    """Create 3×3 intrinsics shared by the whole batch."""
    K = torch.zeros(3, 3, device=fx.device)
    K[0, 0], K[1, 1] = fx, fy
    K[0, 2], K[1, 2] = cx, cy
    K[2, 2] = 1.0
    return K


# --------------------------------------------------------------------------- #
#  Main optimisation                                                          #
# --------------------------------------------------------------------------- #
def optimise_xyz_batch(pred_xyz: torch.Tensor,
                       n_iters: int = 1500,
                       lr: float = 5e-3,
                       verbose: int = 100):
    """
    Parallel optimisation on **B** frames:

    pred_xyz : (B, H, W, 3)  3-D predictions (already in metres or normalised units)
    Returns
    -------
      depth   : (B, H, W)
      K       : (3,3)
      R, t    : (B,3,3), (B,3)
    """
    device               = pred_xyz.device
    B, H, W, _           = pred_xyz.shape
    N                    = H * W                      # pixels per frame
    pred_xyz_flat        = pred_xyz.reshape(B, N, 3) # (B,N,3)

    # -- pixel grid (once, CPU → GPU) ----------------------------------------
    ys, xs               = torch.meshgrid(torch.arange(H),
                                          torch.arange(W),
                                          indexing='ij')
    pix_h                = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1) \
                                .reshape(1, N, 3).to(device)   # (1,N,3)

    # -- parameters ----------------------------------------------------------
    # Depth directly in predicted scale; clamp in forward pass to avoid 0.
    depth = torch.nn.Parameter(pred_xyz_flat.norm(dim=-1))     # (B,N)

    # Shared intrinsics
    fx = torch.nn.Parameter(torch.tensor(W * 1.2, device=device))
    fy = torch.nn.Parameter(torch.tensor(H * 1.2, device=device))
    cx = torch.nn.Parameter(torch.tensor(W / 2,  device=device))
    cy = torch.nn.Parameter(torch.tensor(H / 2,  device=device))

    # Per-frame extrinsics
    rot6d = torch.nn.Parameter(torch.eye(3, device=device)
                               .repeat(B, 1, 1)[:, :, :2]
                               .reshape(B, 6))          # (B,6)
    trans = torch.nn.Parameter(torch.zeros(B, 3, device=device))

    optimiser = torch.optim.Adam([depth, fx, fy, cx, cy, rot6d, trans], lr=lr)

    # -- main loop -----------------------------------------------------------
    for it in range(0, n_iters):
        optimiser.zero_grad()

        # Depth positivity (very mild clamp so gradients flow)
        d = depth.clamp(min=1e-4).unsqueeze(-1)        # (B,N,1)

        K     = build_K(fx, fy, cx, cy)                # (3,3)
        Kinv  = torch.inverse(K).unsqueeze(0)          # (1,3,3)

        # Camera-space pts:   (B,3,3)@(B,3,N) via broadcasting
        cam_pts = torch.matmul(Kinv,                   # (1,3,3)
                               (d * pix_h)             # (B,N,3)
                               .permute(0, 2, 1))      # → (B,3,N)

        # Build per-frame 4×4 cam→world matrices
        R   = rot6d_to_matrix(rot6d)                   # (B,3,3)
        t   = trans.unsqueeze(-1)                      # (B,3,1)

        ones = torch.tensor([0, 0, 0, 1], device=device) \
                    .view(1, 1, 4).repeat(B, 1, 1)     # (B,1,4)

        P_cam2world = torch.cat([torch.cat([R, t], dim=2), ones], dim=1)  # (B,4,4)

        cam_pts_h   = torch.cat([cam_pts, torch.ones(B, 1, N, device=device)], dim=1)
        world_pts   = (P_cam2world @ cam_pts_h)[:, :3]      # (B,3,N)
        world_pts   = world_pts.permute(0, 2, 1)            # (B,N,3)

        loss = (world_pts - pred_xyz_flat).pow(2).mean()
        loss.backward()
        optimiser.step()

        if verbose and it % verbose == 0:
            print(f"[{it:4d}/{n_iters}]  reprojection L2 = {loss.item():.6f}")

    depth_maps = depth.clamp(min=1e-4).detach().reshape(B, H, W)
    K_final    = build_K(fx, fy, cx, cy).detach()
    return depth_maps, K_final, R.detach(), trans.detach()

def depth_to_3d_points(depth_map: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> np.ndarray:
    """Convert depth map to 3D points in world coordinates.
    
    Args:
        depth_map: (H,W) depth values
        K: (3,3) camera intrinsics 
        R: (3,3) rotation matrix
        t: (3,) translation vector
    Returns:
        points: (N,3) 3D points in world coordinates
    """
    H, W = depth_map.shape
    device = depth_map.device
    
    # Get pixel coordinates
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    pix_h = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).to(device)
    
    # Unproject using depth and camera intrinsics
    d = depth_map.unsqueeze(-1)  # Add channel dim
    cam_pts = torch.matmul(torch.inverse(K),
                         (d * pix_h).reshape(-1, 3).T).T  # (N,3)
    
    # Transform to world coordinates  
    world_pts = (R @ cam_pts.T).T + t
    
    return world_pts.cpu().numpy()



def main(
    downsample_factor: int = 1,
    frame_gap: int = 1,
    max_frames: int = 100,
    use_mask: bool = False,
    pkl_dir: str = None,
    monst3r_dir: str = None,
    pointmap_mp4: str = None,
    rgb_mp4: str = None,
    pointmap_npy: str = None,
) -> None:
    # Data loading logic (same as vis.py, with new mp4 support)
    if pointmap_npy is not None and rgb_mp4 is not None:
        pointmap = np.load(pointmap_npy)  # [num_frames, H, W, 3]
        rgb_reader = imageio.get_reader(rgb_mp4, "ffmpeg")
        num_frames = min(pointmap.shape[0], len(rgb_reader), max_frames)
        def PointmapIterator():
            for i in range(num_frames):
                pm = pointmap[i]  # H, W, 3
                rgb = rgb_reader.get_data(i).astype(np.float32) / 255.0  # H, W, 3
                H, W, _ = pm.shape
                yield type('Frame', (), {
                    'pcd': pm.reshape(-1, 3),
                    'pcd_color': rgb.reshape(-1, 3),
                    'rgb': rgb,
                    'T_world_camera': np.eye(4),
                }), f"npy_frame"
    elif pointmap_mp4 is not None and rgb_mp4 is not None:
        pointmap_reader = imageio.get_reader(pointmap_mp4, "ffmpeg")
        rgb_reader = imageio.get_reader(rgb_mp4, "ffmpeg")
        num_frames = 81
        def PointmapIterator():
            for i in range(num_frames):
                pm = pointmap_reader.get_data(i).astype(np.float32) / 255.0  # H, W, 3
                rgb = rgb_reader.get_data(i).astype(np.float32) / 255.0  # H, W, 3
                H, W, _ = pm.shape
                # Flatten for rerun
                yield type('Frame', (), {
                    'pcd': pm.reshape(-1, 3),
                    'pcd_color': rgb.reshape(-1, 3),
                    'rgb': rgb,
                    'T_world_camera': np.eye(4),  # Identity if unknown
                }), f"mp4_frame"
    elif pkl_dir is not None:
        pkl_list = glob.glob(f'{pkl_dir}/*.pkl')
        def PointmapIterator():
            for pkl_path in sorted(pkl_list):
                yield pickle.load(open(pkl_path, 'rb')), pkl_path
    elif monst3r_dir is not None:
        monst3r_list = glob.glob(f'{monst3r_dir}/*/')
        def PointmapIterator():
            for scene_path in monst3r_list:
                yield Monst3RAnno(scene_path, max_frames=max_frames).pointmap, scene_path
    else:
        raise NotImplementedError("No data source provided")

    pointmap_iterator = PointmapIterator()
    if rgb_mp4 is not None:
        raise NotImplementedError("RGB MP4 is not supported yet")
    else:
        for (loader, name) in pointmap_iterator:
            num_frames = min(max_frames, loader.num_frames())
            # Before logging, I want to optimize the pointmap using the function optimise_depth_and_camera
            # First read all pointmap from the pointmap_iterator
            pointmap_list = []
            for i in range(num_frames):
                frame = loader.get_frame(i)
                H, W, _ = frame.rgb.shape
                pointmap_list.append(frame.pcd.reshape(H, W, 3))
            pointmap_list = np.stack(pointmap_list, axis=0)
            # normalize xy to zero centered
            pointmap_list[..., :2] = pointmap_list[..., :2] - 0.5
            pointmap_list = torch.from_numpy(pointmap_list).cuda()
            # Optimize the pointmap
            depth_map, K, R, t = optimise_xyz_batch(pointmap_list)
            updated_pointmap_list = []
            for i in tqdm(range(0, num_frames, frame_gap)):
                frame = loader.get_frame(i)
                position, color = frame.pcd, frame.pcd_color
                # use the new depth_map, K, R, t to unproject to get 3D pointmaps
                current_depth_map = depth_map[i]
                current_R = R[i]
                current_t = t[i]
                # Convert current depth map to 3D points
                position = depth_to_3d_points(current_depth_map, K, current_R, current_t)
                updated_pointmap_list.append(position)
            updated_pointmap = np.stack(updated_pointmap_list, axis=0)
            loader.pcd = updated_pointmap
            save_path = name.replace('.pkl', '_reg.pkl')
            pickle.dump(loader, open(save_path, 'wb'))

if __name__ == "__main__":
    tyro.cli(main)