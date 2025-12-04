from functools import partial
import cv2
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import pandas as pd

from core.dataclass import Pointmap, Video
from core.utils import load_with_cache
from core.utils import load_from_ceph
from core.utils import xy_grid, geotrf

    
class PexelsAnno:
    def __init__(self, video_path, client=None, cache_dir='.cache/', enable_cache=False, caption=None):
        if enable_cache:
            self._load = partial(load_with_cache, client, cache_dir=cache_dir, parse_text_to_float=False)
        else:
            self._load = partial(load_from_ceph, client, cache_dir=cache_dir, parse_text_to_float=False)
        if caption is None:
            caption = self._get_caption(video_path)
        fps = 24
        self.video = Video(
            path=video_path,
            caption=caption,
            fps =fps
        )

    @staticmethod
    def _get_caption(video_path):
        if 'static_1' in video_path:
            df = pd.read_csv(f'./data/caption/{video_path.split("/")[-3]}_with_caption_upload.csv')
            number_str = video_path.split('/')[-2]
        if 'static_2' in video_path:
            df = pd.read_csv(f'./data/caption/{video_path.split("/")[-2]}_with_caption_upload.csv')
            number_str = video_path.split('/')[-1].split('.')[0]
        elif 'dynamic' in video_path:
            df = pd.read_csv(f'./data/caption/{video_path.split("/")[-2]}_with_caption_upload.csv')
            number_str = video_path.split('/')[-1].split('.')[0]
        
        def get_caption_by_number(number):
            result = df.query(f"number == {number}")
            if not result.empty:
                return result.iloc[0]['caption']

        number = int(number_str)
        caption_path = get_caption_by_number(number)
        return caption_path


class Monst3RAnno:
    def __init__(self, anno_dir, client=None, max_frames=None, cache_dir='.cache/', enable_cache=False, caption=None):
        self.anno_dir = anno_dir
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache

        if enable_cache:
            self._load = partial(load_with_cache, client, cache_dir=cache_dir)
        else:
            self._load = partial(load_from_ceph, client, cache_dir=cache_dir)

        # rgb not saved to ceph, load from video
        self.clip_start, self.length = self._get_clip_range(anno_dir)
        video_path = self._get_video_path(anno_dir)
        if not os.path.exists(video_path):
            print(f"Video path {video_path} does not exist!")
        self._video_reader = self._load(video_path)
       
        self.length = min(self.length, max_frames) if max_frames is not None else self.length

        if 'static' in anno_dir:
            rgb, rgb_raw, camera_pose, global_ptmaps = self._load_annotation_static()
            global_ptmaps = global_ptmaps.reshape([self.length, -1, 3])
            self.pointmap = Pointmap(
                pcd=global_ptmaps,    # [T, HxW, 3]
                colors=None,  # [T, HxW, 3]
                rgb=rgb,    # [T, H, W, 3]
                mask= None,  # [T, HxW]
                cams2world=camera_pose, # [T, 4, 4]
                K=None,    # [T, 3, 3]
                depth=None,  # [T, H, W, 3]
            )
        elif 'dynamic' in anno_dir and 'dynamic_3' not in anno_dir:
            rgb, rgb_raw, depth, camera_pose, camera_intrinscis, dynamic_mask = self._load_annotation_dynamic()
            global_ptmaps, colors = self._get_point_cloud(rgb, depth, camera_pose, camera_intrinscis)
            self.pointmap = Pointmap(
                pcd=global_ptmaps,    # [T, HxW, 3]
                colors=colors,  # [T, HxW, 3]
                rgb=rgb,    # [T, H, W, 3]
                mask= dynamic_mask.reshape([self.length, -1]),  # [T, HxW]
                cams2world=camera_pose, # [T, 4, 4]
                K=camera_intrinscis,    # [T, 3, 3]
                depth=depth,  # [T, H, W, 3]
            )
        elif 'dynamic_3' in anno_dir:
            rgb, rgb_raw, depth, camera_pose, camera_intrinscis, dynamic_mask = self._load_annotation_dynamic_3()
            global_ptmaps, colors = self._get_point_cloud(rgb, depth, camera_pose, camera_intrinscis)
            self.pointmap = Pointmap(
                pcd=global_ptmaps,    # [T, HxW, 3]
                colors=colors,  # [T, HxW, 3]
                rgb=rgb,    # [T, H, W, 3]
                mask= dynamic_mask.reshape([self.length, -1]),  # [T, HxW]
                cams2world=camera_pose, # [T, 4, 4]
                K=camera_intrinscis,    # [T, 3, 3]
                depth=depth,  # [T, H, W, 3]
            )


        self.rgb_raw = rgb_raw

        # load video annotation
        self.video = PexelsAnno(video_path=self._get_video_path(anno_dir), client=client, cache_dir=cache_dir, caption=caption).video

    @staticmethod
    def _get_clip_range(anno_dir):
        if os.path.isdir(anno_dir):
            clip_start, clip_end = anno_dir.split('/')[-2].split('.')[0].split('_')[-1].split('-')
            return int(clip_start), int(clip_end) - int(clip_start) + 1
        elif os.path.isfile(anno_dir): # npz file
            filename = os.path.basename(anno_dir)
            name_without_ext = os.path.splitext(filename)[0]
            start_frame, end_frame = name_without_ext.split('-')
            
            if '.png' in name_without_ext:
                start_num = int(start_frame.split('_')[-1].replace('.png', ''))
                end_num = int(end_frame.split('_')[-1].replace('.png', ''))
            else:
                start_num = int(start_frame.split('_')[-1])
                end_num = int(end_frame.split('_')[1])

            clip_start = start_num
            clip_length = end_num - start_num + 1
            
            return clip_start, clip_length

    @staticmethod
    def _get_video_path(anno_dir):
        if 'static_1' in anno_dir:
            video_path = f"./data/raw/static/{anno_dir.split('/')[-3]}/{anno_dir.split('/')[-2]}/images_4"
        elif 'static_2' in anno_dir:
            video_path = f"./data/raw/static/{anno_dir.split('/')[-4]}/{anno_dir.split('/')[-2]}.mp4"
        elif 'dynamic' in anno_dir and 'dynamic_3' not in anno_dir:
            video_path = f"./data/raw/dynamic/{anno_dir.split('/')[-4]}/{anno_dir.split('/')[-3]}.mp4"
        elif 'dynamic_3' in anno_dir:
            video_path = f"./data/raw/dynamic/{anno_dir.split('/')[-3]}/{anno_dir.split('/')[-2]}.mp4"
        return video_path

    @staticmethod
    def _cam_to_RT(poses, xyzw=True):
        num_frames = poses.shape[0]
        poses = np.concatenate(
            [
                # Convert TUM pose to SE3 pose
                Rotation.from_quat(poses[:, 4:]).as_matrix() if not xyzw
                else Rotation.from_quat(np.concatenate([poses[:, 5:], poses[:, 4:5]], -1)).as_matrix(),
                poses[:, 1:4, None],
            ],
            -1,
        )
        poses = poses.astype(np.float32)

        # Convert to homogeneous transformation matrices (ensure shape is (N, 4, 4))
        num_frames = poses.shape[0]
        ones = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (num_frames, 1, 1))
        poses = np.concatenate([poses, ones], axis=1)
        return poses

    @staticmethod
    def _get_point_cloud(rgb, depth, camera_pose, camera_intrinscis):
        T, H, W, _ = rgb.shape
        rgbimg =  torch.from_numpy(rgb)
        focals = torch.from_numpy(camera_intrinscis[:, 0, 0:1])
        cams2world = torch.from_numpy(camera_pose)
        pp = torch.tensor([W//2, H//2])
        pp = torch.stack([pp for _ in range(T)])
        depth = torch.from_numpy(depth)
        
        # maybe cache _grid
        _grid = xy_grid(W, H, device=rgbimg.device)  # [H, W, 2]
        _grid = torch.stack([_grid for _ in range(T)])

        def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
            pp = pp.unsqueeze(1)
            focal = focal.unsqueeze(1)
            assert focal.shape == (len(depth), 1, 1), focal.shape
            assert pp.shape == (len(depth), 1, 2), pp.shape
            assert pixel_grid.shape == depth.shape + (2,), pixel_grid.shape
            depth = depth.unsqueeze(-1)
            pixel_grid = pixel_grid.reshape([pixel_grid.shape[0], -1, pixel_grid.shape[-1]])
            depth = depth.reshape([depth.shape[0], -1, depth.shape[-1]])
            return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)

        rel_ptmaps = _fast_depthmap_to_pts3d(depth, _grid, focals, pp=pp)
        global_ptmaps = geotrf(cams2world, rel_ptmaps)
        colors = rgbimg.reshape([rgbimg.shape[0], -1, rgbimg.shape[-1]])

        return global_ptmaps.numpy(), colors.numpy()


    def _load_annotation_dynamic(self):
        pred_traj = self._load(self.anno_dir + f'pred_traj.txt')
        pred_intrinsics = self._load(self.anno_dir + f'pred_intrinsics.txt')

        cam_pose_list = []
        rgb_list = []
        rgb_raw_list = []
        depth_list = []
        mask_list = []
        cam_intrinscis_list = []

        for t in range(self.length):
            # load depth
            depth = self._load(self.anno_dir + f'frame_{t:04d}.npy')
            depth_list.append(depth)
            H, W = depth.shape[0], depth.shape[1]

            # load rgb
            if isinstance(self._video_reader, np.ndarray):
                rgb_raw = self._video_reader[self.clip_start + t, ...]
            elif isinstance(self._video_reader, list):
                rgb_raw = self._video_reader[self.clip_start + t]
            else:
                rgb_raw = self._video_reader.get_data(self.clip_start + t)
            rgb = cv2.resize(rgb_raw, (W, H))
            rgb = rgb.astype(np.float32) / 255
            rgb_raw = rgb_raw.astype(np.float32) / 255
            rgb_list.append(rgb)
            rgb_raw_list.append(rgb_raw)

            # load dynamic mask
            mask = self._load(self.anno_dir + f'dynamic_mask_{t}.png')
            mask_list.append(mask)

            # load camera
            cam_pose_list.append(pred_traj[t])
            cam_intrinscis_list.append(pred_intrinsics[t])

        cam_pose_list = np.stack(cam_pose_list)     # [T, 7]
        cam_intrinscis_list = np.stack(cam_intrinscis_list)     # [T, 9]
        rgb_list = np.stack(rgb_list)       # [T, H, W ,3]
        rgb_raw_list = np.stack(rgb_raw_list)       # [T, H_raw, W_raw, 3]
        depth_list = np.stack(depth_list)     # [T, H, W]
        mask_list = np.stack(mask_list)     # [T, H, W]

        cam_pose_list = self._cam_to_RT(cam_pose_list)  # [T, 4, 4]
        cam_intrinscis_list = cam_intrinscis_list.reshape([-1, 3, 3])    # [T, 3, 3]

        return rgb_list, rgb_raw_list, depth_list, cam_pose_list, cam_intrinscis_list, mask_list
    
    def _load_annotation_dynamic_3(self):
        # Load all data from npz file
        npz_path = self.anno_dir
        npz_data = np.load(npz_path, allow_pickle=True)
        
        # Extract data from npz file
        images = npz_data['images']           # [T, H, W, 3] or [T, H, W]
        depths = npz_data['depths']           # [T, H, W]
        intrinsics = npz_data['intrinsic']    # Could be scalar, [3,3] or [T, 3, 3]
        cam_c2w = npz_data['cam_c2w']         # [T, 4, 4] camera-to-world transformation matrices
        
        # Ensure images are 3-channel and in correct range [0, 1]
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
        
        # Convert 2D images (single channel) to 3-channel
        if len(images.shape) == 3:
            images = np.stack([images] * 3, axis=-1)
        
        # Set sequence length based on number of images
        self.length = images.shape[0]
        
        # Handle different intrinsic matrix formats
        if intrinsics.ndim == 0:
            # If scalar, tile to all frames
            intrinsics = np.tile(intrinsics, (self.length, 1, 1))
        elif intrinsics.ndim == 2:  # Single [3, 3] matrix
            intrinsics = np.tile(intrinsics[np.newaxis, :, :], (self.length, 1, 1))
        elif intrinsics.ndim == 3 and intrinsics.shape[0] == 1:  # [1, 3, 3]
            intrinsics = np.tile(intrinsics, (self.length, 1, 1))
        
        # Ensure depth sequence matches image sequence length
        if depths.shape[0] != self.length:
            # Truncate or adjust if mismatch
            depths = depths[:self.length]
            cam_c2w = cam_c2w[:self.length]
        
        # Get original image dimensions
        H_raw, W_raw = images.shape[1:3]
        
        # Get depth map dimensions
        H, W = depths.shape[1:3]
        
        # Initialize lists for each data type
        rgb_list = []
        rgb_raw_list = []
        depth_list = []
        cam_pose_list = []
        cam_intrinsics_list = []
        
        for t in range(self.length):
            # Depth map
            depth = depths[t]
            depth_list.append(depth)
            
            # RGB image (original resolution)
            rgb_raw = images[t]
            rgb_raw_list.append(rgb_raw)
            
            # Resize RGB if needed to match depth dimensions
            if H != H_raw or W != W_raw:
                rgb = cv2.resize(rgb_raw, (W, H))
            else:
                rgb = rgb_raw.copy()
            rgb_list.append(rgb)
            
            # Camera pose (camera-to-world matrix)
            cam_pose_list.append(cam_c2w[t])
            
            # Camera intrinsics
            cam_intrinsics_list.append(intrinsics[t])
        
        # Stack all data into numpy arrays
        cam_pose_list = np.stack(cam_pose_list)          # [T, 4, 4]
        cam_intrinsics_list = np.stack(cam_intrinsics_list)  # [T, 3, 3]
        rgb_list = np.stack(rgb_list)                   # [T, H, W, 3]
        rgb_raw_list = np.stack(rgb_raw_list)           # [T, H_raw, W_raw, 3]
        depth_list = np.stack(depth_list)               # [T, H, W]
        
        # Note: The npz file doesn't contain mask data
        # Create zero mask as placeholder (or load from separate source)
        mask_list = np.zeros_like(depth_list, dtype=np.uint8)
        
        return rgb_list, rgb_raw_list, depth_list, cam_pose_list, cam_intrinsics_list, mask_list

    def _load_annotation_static(self):
        npz_path = self.anno_dir
        npz_data = np.load(npz_path, allow_pickle=True)
        data = {}
        for key in npz_data.files:
            if npz_data[key].shape == ():
                content = npz_data[key].item()
                for k, v in content.items():
                    data[k] = v
        rgb_list = []
        rgb_raw_list = []
        for t in range(self.length):
            H, W = data['pts3d'].shape[1], data['pts3d'].shape[2]
            # load rgb
            if isinstance(self._video_reader, np.ndarray):
                rgb_raw = self._video_reader[self.clip_start + t, ...]
            elif isinstance(self._video_reader, list):
                rgb_raw = self._video_reader[self.clip_start + t]
            else:
                rgb_raw = self._video_reader.get_data(self.clip_start + t)
            rgb = cv2.resize(rgb_raw, (W, H))
            rgb = rgb.astype(np.float32) / 255
            rgb_raw = rgb_raw.astype(np.float32) / 255

            rgb_list.append(rgb)
            rgb_raw_list.append(rgb_raw)

        rgb_list = np.stack(rgb_list)       # [T, H, W ,3]
        rgb_raw_list = np.stack(rgb_raw_list)       # [T, H_raw, W_raw, 3]
     
        return rgb_list, rgb_raw_list, data['poses'].astype(np.float32), data['pts3d'].astype(np.float32)