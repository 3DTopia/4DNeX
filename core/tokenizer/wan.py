import torch
from diffusers import AutoencoderKLWan

from core.tokenizer.base import BaseTokenizer


class WanTokenizer(BaseTokenizer):

    def __init__(self, model_path, dtype='bfloat16', device='cuda'):
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.model = AutoencoderKLWan.from_pretrained(model_path, torch_dtype=self.dtype).to(self.device)

    @torch.no_grad()
    def encode(self, frames):
        """encode frames, [0, 1]"""
        frames = frames.transpose(0,3,1,2)
        frames = frames * 2 - 1
        frames_tensor = torch.from_numpy(frames).to(self.device).permute(1, 0, 2, 3).unsqueeze(0).to(self.dtype)
        # encoded_frames = self.model.encode(frames_tensor)[0].sample()
        latent = self.model.encode(frames_tensor)[0].sample()

        # Get latent scaling factor from VAE config
        latents_mean = torch.tensor(self.model.config.latents_mean).view(1, self.model.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
        latents_std = 1.0 / torch.tensor(self.model.config.latents_std).view(1, self.model.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
        encoded_frames = (latent - latents_mean) * latents_std
        return encoded_frames, {}

    @torch.no_grad()
    def decode(self, latents):
        """decode frames, [0, 1]"""
        latents_mean = (
            torch.tensor(self.model.config.latents_mean)
            .view(1, self.model.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.model.config.latents_std).view(1, self.model.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        encoded_frames = latents.to(self.device).to(self.dtype)
        decoded_frames = self.model.decode(encoded_frames).sample
        decoded_frames = decoded_frames.to(dtype=torch.float32)
        decoded_frames = decoded_frames[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
        decoded_frames = decoded_frames / 2 + 0.5
        return decoded_frames