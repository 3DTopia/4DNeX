import glob
import time
import pickle
import itertools
from pathlib import Path

import numpy as np
import tyro
from tqdm.auto import tqdm

import rerun as rr
import imageio

from core.dataset import PointmapDataset
from core.annotation import Monst3RAnno

def log_image_plane_outline(rr, name, K, T_world_camera, H, W):
    z = 1.0
    corners_px = np.array([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H],
        [0, 0],  # Close the loop
    ])
    Kinv = np.linalg.inv(K)
    corners_cam = []
    for u, v in corners_px:
        vec = np.array([u, v, 1.0])
        xyz = Kinv @ vec
        xyz = xyz / xyz[2] * z
        corners_cam.append(xyz)
    corners_cam = np.stack(corners_cam, axis=0)
    R = T_world_camera[:3, :3]
    t = T_world_camera[:3, 3]
    corners_world = (R @ corners_cam.T).T + t
    rr.log(
        f"{name}/image_frustum_outline",
        rr.LineStrips3D([corners_world])
    )

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
    rr_recording: str = "pointmap_log",
) -> None:
    rr.init(rr_recording)

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
        pkl_list = glob.glob(f'{pkl_dir}/*_reg.pkl')
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
        # Each iteration yields a single frame and name, so just log it once
        idx = 0
        for (frame, name) in pointmap_iterator:
            position, color = frame.pcd, frame.pcd_color
            rgb = frame.rgb
            H, W, _ = rgb.shape
            position = position.reshape(H, W, 3)[::downsample_factor, ::downsample_factor].reshape([-1, 3])
            color = color.reshape(H, W, 3)[::downsample_factor, ::downsample_factor].reshape([-1, 3])
            rr.set_time_sequence("frame", idx)  # Use frame index from name
            # Log point cloud
            rr.log(
                f"{name}/point_cloud",
                rr.Points3D(
                    positions=position,
                    colors=(color * 255).astype(np.uint8),
                ),
            )
            # Log RGB image (optional)
            rr.log(
                f"{name}/rgb_image",
                rr.Image(rgb),
            )
            # Log camera frustum (if possible)
            # rerun does not have a direct frustum primitive, but we can log camera pose as transform
            rr.log(
                f"{name}/camera",
                rr.Transform3D(
                    translation=frame.T_world_camera[:3, 3],
                    mat3x3=frame.T_world_camera[:3, :3],
                ),
            )
            if use_mask and hasattr(frame, 'fg_pcd'):
                # Foreground
                rr.log(
                    f"{name}/fg_point_cloud",
                    rr.Points3D(
                        positions=frame.fg_pcd,
                        colors=(frame.fg_pcd_color * 255).astype(np.uint8),
                    ),
                )
                # Background
                rr.log(
                    f"{name}/bg_point_cloud",
                    rr.Points3D(
                        positions=frame.bg_pcd,
                        colors=(frame.bg_pcd_color * 255).astype(np.uint8),
                    ),
                )
            idx += 1
    else:
        for (loader, name) in pointmap_iterator:
            num_frames = min(max_frames, loader.num_frames())
            for i in tqdm(range(0, num_frames, frame_gap)):
                frame = loader.get_frame(i)
                position, color = frame.pcd, frame.pcd_color
                rgb = frame.rgb
                H, W, _ = rgb.shape
                position = position.reshape(H, W, 3)[::downsample_factor, ::downsample_factor].reshape([-1, 3])
                color = color.reshape(H, W, 3)[::downsample_factor, ::downsample_factor].reshape([-1, 3])
                rr.set_time_sequence("frame", i)
                # Log point cloud
                rr.log(
                    f"{name}/point_cloud",
                    rr.Points3D(
                        positions=position,
                        colors=(color * 255).astype(np.uint8),
                    ),
                )
                # Log RGB image (optional)
                rr.log(
                    f"{name}/rgb_image",
                    rr.Image(rgb),
                )
                # Log camera frustum (if possible)
                # rerun does not have a direct frustum primitive, but we can log camera pose as transform
                rr.log(
                    f"{name}/camera",
                    rr.Transform3D(
                        translation=frame.T_world_camera[:3, 3],
                        mat3x3=frame.T_world_camera[:3, :3],
                    ),
                )

                # log_image_plane_outline(rr, name, frame.K, frame.T_world_camera, H, W)

                if use_mask and hasattr(frame, 'fg_pcd'):
                    # Foreground
                    rr.log(
                        f"{name}/fg_point_cloud",
                        rr.Points3D(
                            positions=frame.fg_pcd,
                            colors=(frame.fg_pcd_color * 255).astype(np.uint8),
                        ),
                    )
                    # Background
                    rr.log(
                        f"{name}/bg_point_cloud",
                        rr.Points3D(
                            positions=frame.bg_pcd,
                            colors=(frame.bg_pcd_color * 255).astype(np.uint8),
                        ),
                    )
            print(f"Finished logging {name}. Open rerun viewer to inspect.")
            # break
    rr.save(rr_recording)

if __name__ == "__main__":
    tyro.cli(main)
