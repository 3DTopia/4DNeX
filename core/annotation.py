from functools import partial
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

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
            caption = self._load(self._get_caption_path(video_path))[0]
        fps = 24
        self.video = Video(
            path=video_path,
            caption=caption,
            fps =fps
        )
    @staticmethod
    def _get_caption_path(video_path):
        caption_path = video_path.replace('4DGen-Dataset/Human_Raw_Data', '4DGen-Dataset/Human_Raw_Data/sensetime/caption').replace('.mp4', '.txt')
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

        # rgb not saved to ceph, load from video if using s3 path
        if "s3://" in self.anno_dir:
            self.clip_start, self.length = self._get_clip_range(anno_dir)
            self._video_reader = self._load(self._get_video_path(anno_dir))
        else:
            self.clip_start = -1
            self.length = len(self._load(self.anno_dir + f'pred_traj.txt'))

        self.length = min(self.length, max_frames) if max_frames is not None else self.length

        rgb, rgb_raw, depth, camera_pose, camera_intrinscis, dynamic_mask = self._load_annotation()
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
        if "s3://" in self.anno_dir:
            self.video = PexelsAnno(video_path=self._get_video_path(anno_dir), client=client, cache_dir=cache_dir, caption=caption).video
        else:
            self.video = None


    @staticmethod
    def _get_clip_range(anno_dir):
        clip_start, clip_end = anno_dir.split('/')[-2].split('.')[0].split('_')[-1].split('-')
        return int(clip_start), int(clip_end) - int(clip_start) + 1

    @staticmethod
    def _get_video_path(anno_dir):
        if "vbench" in anno_dir:
            video_path = anno_dir.replace("vbench", "vbench_raw_video")
            video_path = video_path.split('/clip')[0] + '.mp4'
        else:
            uid_frameid = anno_dir.split('/')[-3]
            uid = uid_frameid.split('-')[0]
            pexel_name = anno_dir.split('/')[-4]
            video_path = f"pvgen:s3://4DGen-Dataset/Human_Raw_Data/pexels/{pexel_name}/{uid}/{uid_frameid}.mp4"
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


    def _load_annotation(self):
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
            if "s3://" in self.anno_dir:
                if isinstance(self._video_reader, np.ndarray):
                    rgb_raw = self._video_reader[self.clip_start + t, ...]
                else:
                    rgb_raw = self._video_reader.get_data(self.clip_start + t)
                rgb = cv2.resize(rgb_raw, (W, H))
                # save rgb frames (optional)
                if self.enable_cache:
                    cv2.imwrite(self.cache_dir + self.anno_dir.split('s3://')[-1] + f'frame_{t:04d}.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                rgb = rgb.astype(np.float32) / 255
                rgb_raw = rgb_raw.astype(np.float32) / 255
            else:
                rgb = self._load(self.anno_dir + f'frame_{t:04d}.png')
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
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
