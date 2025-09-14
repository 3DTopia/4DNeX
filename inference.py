import argparse
import os
import torch
import numpy as np
import imageio
import pickle
from core.inference.wan import generate_video
from core.dataclass import Pointmap
from core.tokenizer.wan import WanTokenizer

model_path = 'pretrained/Wan2.1-I2V-14B-480P-Diffusers/vae/'
tokenizer = WanTokenizer(model_path=model_path)

def save_pointmap(latents, save_path, image_path=None, mode='xyz'):
    latents = latents[None]
    # apply special denormalize to latent pointmap only
    encoded_pm_mean = -0.13
    encoded_pm_std = 1.70
    latents[:, :, :, :, latents.shape[4]//2:] = latents[:, :, :, :, latents.shape[4]//2:] * encoded_pm_std + encoded_pm_mean
    pointmap = tokenizer.decode(latents)
    if '.mp4' in save_path:
        mp4_save_path = save_path
    else:
        _, ext = os.path.splitext(save_path)
        mp4_save_path = save_path.replace(ext, ".mp4")
    imageio.mimwrite(mp4_save_path,(pointmap*255).clip(0, 255).astype(np.uint8),fps=24)
    
    pm = Pointmap()
    if mode == 'xyzrgb':
        W =  pointmap.shape[2] // 2
        rgb = pointmap[..., :W, :]
        pointmap = pointmap[..., W:, :]
    pm.init_dummy(pointmap.shape[0], pointmap.shape[1], pointmap.shape[2])
    pointmap = pointmap.reshape(*pm.pcd.shape)
    pm.pcd = pointmap
    if mode == 'xyzrgb':
        pm.rgb = rgb.clip(min=0, max=1)
        pm.colors = pm.rgb.reshape(*pm.colors.shape)
    elif image_path is not None:
        rgb = imageio.imread(image_path) / 255.
        pm.rgb = np.stack([rgb for _ in range(pm.rgb.shape[0])], 0)
        pm.colors = pm.rgb.reshape(*pm.colors.shape)
        
    pickle.dump(pm, open(save_path, 'wb'))

def main(args):
    prompt_list = []
    with open(args.prompt, 'r') as f:
        for line in f.readlines():
            prompt_list.append(line.strip())

    image_list = []
    with open(args.image, 'r') as f:
        for line in f.readlines():
            image_list.append(line.strip())
    
    assert len(prompt_list) == len(image_list)

    os.makedirs(args.out, exist_ok=True)
    for i in range(len(prompt_list)):
        prompt, image_path = prompt_list[i], image_list[i]
        suffix = 'POINTMAP_STYLE.'
        prompt = prompt + ' ' + suffix
        output_path = os.path.join(args.out, f'{i:05d}.mp4')
        if args.idx==-1 or i==args.idx:
            latent = generate_video(
                prompt=prompt,
                image_or_video_path=image_path,
                model_path='pretrained/Wan2.1-I2V-14B-480P-Diffusers',
                sft_path=args.sft_path,
                lora_path=args.lora_path,
                lora_rank=args.lora_rank,
                output_path=output_path,
                num_frames=49,
                width=720 * 2,
                height=480,
                generate_type=args.type,
                num_inference_steps=50,
                guidance_scale=5.0,
                fps=24,
                num_videos_per_prompt=1,
                dtype=torch.bfloat16,
                seed=42,
                mode=args.mode
            )
            save_pointmap(latent, output_path.replace('.mp4', '.pkl'), image_path, args.mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Wan")
    parser.add_argument("--prompt", type=str, required=True, help="prompt list")
    parser.add_argument("--image", type=str, required=True, help="image list")
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--sft_path", type=str, default=None, help="The path of the SFT weights to be used")
    parser.add_argument("--out", type=str, default="results/output", help="The path save generated video")
    parser.add_argument("--mode", type=str, default="xyzrgb", help="xyz or xzyrgb")
    parser.add_argument("--type", type=str, default="condpm-i2dpm", help="i2dpm or condpm-i2dpm")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=64, help="The rank of the LoRA weights to be used")
    args = parser.parse_args()
    main(args)
