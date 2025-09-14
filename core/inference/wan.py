import logging
from typing import Literal, Optional
import os
import shutil
import numpy as np

import torch
from PIL import Image

from transformers import CLIPVisionModel
from diffusers import (
    WanImageToVideoPipeline,
    WanPipeline,
)
from diffusers.utils import export_to_video, load_image, load_video

logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    "wan2.1-i2v-14b-480p-diffusers": (480, 720),
    "wan2.1-i2v-14b-720p-diffusers": (720, 1280),
}


def generate_video(
    prompt: str,
    model_path: str,
    sft_path: str = None,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v",],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
    mode: str = 'xyz',
):
    """
    Generates a video based on the given prompt and saves it to the specified path.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=torch.float32)

    if sft_path:
        print('loading SFT weight')
        if generate_type == "i2dpm":
            raise NotImplementedError
        elif generate_type == "i2vwbw-demb-samerope":
            from core.finetune.models.wan_i2v.demb_samerope_trainer import WanTransformer3DModelDembSameRope
            transformer = WanTransformer3DModelDembSameRope.from_pretrained(sft_path, torch_dtype=dtype)
            assert lora_path is not None, "Lora path is required for i2vwbw-demb-samerope"
            learnable_domain_embeddings = torch.load(os.path.join(lora_path, "learnable_domain_embeddings.pt"))
            transformer.learnable_domain_embeddings.data = learnable_domain_embeddings.to(transformer.device, transformer.dtype)
            print(f"Loaded learnable_domain_embeddings from {lora_path}")
        else:
            from diffusers import WanTransformer3DModel
            config_path = os.path.join(sft_path, 'config.json')
            src_config_path = os.path.join(model_path, 'transformer', 'config.json')
            if not os.path.exists(config_path):
                shutil.copyfile(src_config_path, config_path)
            transformer = WanTransformer3DModel.from_pretrained(sft_path, torch_dtype=dtype)
    else:
        transformer = None

    if generate_type == "i2v":
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_path, 
            image_encoder=image_encoder,
            transformer=transformer,
            torch_dtype=dtype
        )
        image = load_image(image=image_or_video_path)
        if mode == 'xyzrgb':
            twidth, theight = image.size
            left_half = image.crop((twidth // 4, 0, (twidth * 3)// 4, theight))
            # Create a new image with the same size
            new_image = Image.new("RGB", (twidth, theight))
            # Paste the left half twice
            new_image.paste(left_half, (0, 0))
            new_image.paste(left_half, (twidth // 2, 0))
            image = new_image
    elif generate_type == "t2v":
        pipe = WanPipeline.from_pretrained(model_path, torch_dtype=dtype)
    elif generate_type == "i2vhbh":
        from core.finetune.models.wan_i2v.sft_trainer import WanHBHImageToVideoPipeline
        pipe = WanHBHImageToVideoPipeline.from_pretrained(model_path, image_encoder=image_encoder, transformer=transformer, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "i2vwbw-demb-samerope":
        from core.finetune.models.wan_i2v.demb_samerope_trainer import WanSameRopeWBWImageToVideoPipeline
        pipe = WanSameRopeWBWImageToVideoPipeline.from_pretrained(model_path, image_encoder=image_encoder, transformer=transformer, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    else:
        raise NotImplementedError

    max_area = 480 * 720
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
        
    # If you're using with lora, add this code
    if lora_path:
        print('loading lora')
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"], lora_scale=0.5)

    pipe.to("cuda")

    # 4. Generate the video frames based on the prompt.
    if generate_type == "i2v" or generate_type == "i2vhbh" or generate_type == "i2vwbw-demb-samerope":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
            output_type="latent"
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
            output_type="latent"
        ).frames[0]
    elif generate_type == "i2dpm":
        video_generate = pipe(
            height=height,
            width=width,
            prompt=prompt,
            image=image,
            # The path of the image, the resolution of video will be the same as the image for CogVideoX1.5-5B-I2V, otherwise it will be 720 * 480
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=num_frames,  # Number of frames to generate
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
            output_type="latent"
        ).frames
    else:
        raise NotImplementedError

    return video_generate