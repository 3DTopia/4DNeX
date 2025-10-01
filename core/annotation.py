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
        if 'static' in video_path:
            df = pd.read_csv(f'/share/project/cwm/tianqi.liu/workspace/zhaoxi/4DNeX-main/dataset/caption/{video_path.split("/")[-3]}_with_caption_upload.csv')
            number_str = video_path.split('/')[-2]
        elif 'dynamic' in video_path:
            df = pd.read_csv(f'/share/project/cwm/tianqi.liu/workspace/zhaoxi/4DNeX-main/dataset/caption/{video_path.split("/")[-2]}_with_caption_upload.csv')
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
        elif 'dynamic' in anno_dir:
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
            
            start_num = int(start_frame.split('_')[-1].replace('.png', ''))
            end_num = int(end_frame.split('_')[-1].replace('.png', ''))
            
            clip_start = start_num
            clip_length = end_num - start_num + 1
            
            return clip_start, clip_length

    @staticmethod
    def _get_video_path(anno_dir):
        if 'static' in anno_dir:
            video_path = f"/share/project/cwm/tianqi.liu/workspace/zhaoxi/4DNeX-main/dataset/raw/static/{anno_dir.split('/')[-3]}/{anno_dir.split('/')[-2]}/images_4"
        elif 'dynamic' in anno_dir:
            video_path = f"/share/project/cwm/tianqi.liu/workspace/zhaoxi/4DNeX-main/dataset/raw/dynamic/{anno_dir.split('/')[-4]}/{anno_dir.split('/')[-3]}.mp4"
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

    def _load_annotation_static(self):
        npz_path = self.anno_dir
        npz_data = np.load(npz_path, allow_pickle=True)
        data = {}
        for key in npz_data.files:
            if npz_data[key].shape == ():
                content = npz_data[key].item()
                for k, v in content.items():
                    data[k] = v
                    print(f"  {k}: 类型={type(v)}, 形状={getattr(v, 'shape', '无形状')}")
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