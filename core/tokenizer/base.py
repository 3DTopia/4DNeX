import numpy as np
from abc import abstractmethod

class BaseTokenizer:
    @abstractmethod
    def __init__(self, model_path, torch_dtype, device):
        pass
    @staticmethod
    def normalize(frames, mean, std):
        return (frames - mean) / std
    @staticmethod
    def denormalize(frames, mean, std):
        return frames * std + mean
    
    @staticmethod
    def normalize_instance(frames, range_min=-1):
        channel_min, channel_max = frames.min(1, keepdims=True), frames.max(1, keepdims=True)
        center = (channel_min + channel_max) / 2
        scale = (channel_max - channel_min) / 2
        scale = scale.max()     # same scale for all axes
        frames = (frames - center) / scale   # normalize to [-1, 1]
        if range_min == 0:
            frames = (frames + 1) * 0.5
        return frames, center, scale
    
    @staticmethod
    def denormalize_instance(frames, center, scale, range_min=-1):
        if range_min == 0:
            frames = frames * 2 - 1
        frames = frames * scale + center
        return frames
    @staticmethod
    def sigmoid(frames, scale=1):
        frames = 1 / (1+np.exp(frames * -scale))
        return frames
    @staticmethod
    def inverse_sigmoid(frames, scale=1):
        frames = -1 * np.log(1/ frames - 1) / scale
        return frames