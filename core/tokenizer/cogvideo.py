import torch
from diffusers import AutoencoderKLCogVideoX

from core.tokenizer.base import BaseTokenizer


class CogvideoTokenizer(BaseTokenizer):

    def __init__(self, model_path, dtype='bfloat16', device='cuda'):
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.model = AutoencoderKLCogVideoX.from_pretrained(model_path, torch_dtype=self.dtype).to(self.device)
        self.vae_scaling_factor_image = self.model.config.scaling_factor

    @torch.no_grad()
    def encode(self, frames):
        """encode frames, [0, 1]"""
        # model.enable_slicing()
        # model.enable_tiling()
        frames = frames.transpose(0,3,1,2)
        frames = frames * 2 - 1
        frames_tensor = torch.from_numpy(frames).to(self.device).permute(1, 0, 2, 3).unsqueeze(0).to(self.dtype)
        encoded_frames = self.model.encode(frames_tensor)[0].sample()
        return encoded_frames, {}

    @torch.no_grad()
    def decode(self, latents):
        """decode frames, [0, 1]"""
        # https://github.com/huggingface/diffusers/blame/560fb5f4d65b8593c13e4be50a59b1fd9c2d9992/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py#L418C19-L418C19
        latents = 1 / self.vae_scaling_factor_image * latents
        encoded_frames = latents.to(self.device).to(self.dtype)
        decoded_frames = self.model.decode(encoded_frames).sample
        decoded_frames = decoded_frames.to(dtype=torch.float32)
        decoded_frames = decoded_frames[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        decoded_frames = decoded_frames / 2 + 0.5
        return decoded_frames