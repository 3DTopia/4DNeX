import numpy as np
from dataclasses import dataclass

@dataclass
class Video:
    path: str
    caption: str
    fps: int

@dataclass
class Frame:
    rgb: np.ndarray
    pcd: np.ndarray
    pcd_color: np.ndarray
    T_world_camera: np.ndarray
    K: np.ndarray
    fg_pcd: np.ndarray
    fg_pcd_color: np.ndarray
    bg_pcd: np.ndarray
    bg_pcd_color: np.ndarray

@dataclass
class Pointmap:
    pcd: np.ndarray = None
    colors: np.ndarray = None
    rgb: np.ndarray = None
    mask: np.ndarray = None
    cams2world: np.ndarray = None
    K:  np.ndarray = None
    depth: np.ndarray = None

    def num_frames(self):
        return self.pcd.shape[0]

    def init_dummy(self, F, H, W):
        self.pcd = np.zeros([F, H*W, 3])
        self.colors = np.zeros([F, H*W, 3])
        self.rgb = np.zeros([F, H, W, 3])
        self.mask = np.zeros([F, H*W])
        self.mask[:, :H*W//2] = 1
        self.cams2world = np.tile(np.eye(4), (F, 1, 1))
        self.K = np.zeros([F, 3, 3])
        self.depth = np.zeros([F, H, W])

    def get_frame(self, t):
        mask = self.mask[t]
        return Frame(
            rgb=self.rgb[t],
            pcd=self.pcd[t],
            pcd_color=self.colors[t],
            T_world_camera=self.cams2world[t],
            K=self.K[t],
            fg_pcd=self.pcd[t][mask==1],
            fg_pcd_color=self.colors[t][mask==1],
            bg_pcd=self.pcd[t][mask==0],
            bg_pcd_color=self.colors[t][mask==0],
        )

    def get_resolution(self):
        return self.rgb.shape[1], self.rgb.shape[2]