import os
import hashlib
import numpy as np
import imageio
import torch
from pathlib import Path
import torchvision.transforms.functional as F
import torch.nn.functional as nnf
import argparse
from core.dataset import PointmapDataset
from transformers import AutoTokenizer, UMT5EncoderModel, CLIPVisionModel, CLIPImageProcessor
from diffusers import AutoencoderKLWan
from safetensors.torch import save_file
import PIL


def transform_pointclouds_to_first_camera(pointclouds, cams2world):
    """
    pointclouds: [F, N, 3] in world coordinates
    cams2world: [F, 4, 4] original camera-to-world matrices
    Returns: [F, N, 3] pointclouds in the new coordinate system (first camera as origin)
    """
    # Invert the first pose
    world2first = np.linalg.inv(cams2world[0])  # [4, 4]
    
    # Convert pointclouds to homogeneous coordinates
    F, N, _ = pointclouds.shape
    ones = np.ones((F, N, 1), dtype=pointclouds.dtype)
    points_homo = np.concatenate([pointclouds, ones], axis=-1)  # [F, N, 4]
    
    # Apply the transformation
    # (world2first @ point_homo.T).T for each frame
    points_transformed = np.einsum('ij,fkj->fki', world2first, points_homo)  # [F, N, 4]
    
    # Drop the homogeneous coordinate
    return points_transformed[..., :3]

def split_list(datalist, num_segments):
    """Splits list datalist into num_segments continuous segments with balanced load."""
    n = len(datalist)
    segment_size = n // num_segments
    remainder = n % num_segments  # Extra elements to distribute

    segments = []
    start = 0
    for i in range(num_segments):
        extra = 1 if i < remainder else 0  # Distribute remainder among the first few segments
        end = start + segment_size + extra
        segments.append(datalist[start:end])
        start = end

    return segments


def encode_video(video, vae):
    """Encode video using Wan VAE."""
    # Move to VAE's device and dtype
    # Assume video is in shape [B, F, H, W, C]
    # Check video values are approximately in [-1, 1] range with small tolerance
    assert torch.max(torch.abs(video)) <= 1.05, "Input video values must be approximately in range [-1, 1]. Now it is " + str(torch.max(torch.abs(video)))
    video = video.to(vae.device, dtype=vae.dtype)
    video = video.permute(0, 4, 1, 2, 3)
    # Encode using VAE
    with torch.no_grad():
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample()
        # Get latent scaling factor from VAE config
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latent.device, latent.dtype)
        latent = (latent - latents_mean) * latents_std
    return latent


def encode_text(prompt, tokenizer, text_encoder, max_text_seq_length, device):
    """Encode text using UMT5 encoder."""
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_text_seq_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_text_seq_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )
    return prompt_embeds


def encode_image(image, image_processor, image_encoder, device):
    """Encode image using CLIP vision encoder."""
    image = image_processor(images=image, return_tensors="pt").to(device)
    image_embeds = image_encoder(**image, output_hidden_states=True)
    return image_embeds.hidden_states[-2]


def main(args):
    data_dir = Path(args.out)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(data_dir / "videos", exist_ok=True)
    # os.makedirs(data_dir / "video_latents", exist_ok=True)
    os.makedirs(data_dir / "pointmap", exist_ok=True)
    os.makedirs(data_dir / "pointmap_latents", exist_ok=True)
    # os.makedirs(data_dir / "datalist", exist_ok=True)
    # os.makedirs(data_dir / "prompts", exist_ok=True)
    # os.makedirs(data_dir / "center", exist_ok=True)
    # os.makedirs(data_dir / "scale", exist_ok=True)
    os.makedirs(data_dir / "first_frames", exist_ok=True)  # Add directory for first frames
    
    max_frames = 81     # Wan required frame length
    resolution_out = (args.resolution_h, args.resolution_w)     # Wan required resolution
    resolution_crop = resolution_out

    # Create cache directories
    cache_dir = data_dir / "cache"
    train_resolution_str = f"{max_frames}x{args.resolution_h}x{args.resolution_w}"
    video_latent_dir = cache_dir / "video_latent" / "wan-i2v" / train_resolution_str
    prompt_embeddings_dir = cache_dir / "prompt_embeddings"
    video_latent_dir.mkdir(parents=True, exist_ok=True)
    prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset samples
    # pexel_datalist = []
    # with open(args.datalist, 'r') as f:
    #     for line in f.readlines():
    #         pexel_datalist.append(line.strip())
        # Generate datalist from directory
    print(f"Generating datalist from directory: {args.data_dir}")
    pexel_datalist = generate_datalist_from_directory(args.data_dir)
    print(f"Found {len(pexel_datalist)} samples")
    
    # Save the generated datalist for reference
    datalist_path = data_dir / "generated_datalist.txt"
    with open(datalist_path, 'w') as f:
        f.write('\n'.join(pexel_datalist))
    print(f"Saved generated datalist to {datalist_path}")

    pexel_datalist_segment = split_list(pexel_datalist, num_segments=args.num_tasks)[args.task_idx]
    print(f'{args.task_idx}/{args.num_tasks} task | {len(pexel_datalist_segment)}/{len(pexel_datalist)} samples')
    
    # Initialize dataset
    cache_dir_tmp = f'.cache/{args.task_idx:05d}/'
    train_dataset_iterator = iter(
            PointmapDataset(
                datalist = pexel_datalist_segment,
                max_frames = max_frames,
                s3_conf_path = '~/petreloss.conf',
                debug=False,
                random_shuffle=False,
                cache_dir=cache_dir_tmp,
                skip_invalid=False
        )
    )

    # Initialize models and processors
    model_path = args.model_path
    
    # 1. VAE for video encoding
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae")
    vae.to(args.device)
    
    # 2. Text encoder and tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")
    text_encoder.to(args.device)
    
    # 3. Image encoder and processor
    image_processor = CLIPImageProcessor.from_pretrained(model_path, subfolder="image_processor")
    image_encoder = CLIPVisionModel.from_pretrained(model_path, subfolder="image_encoder")
    image_encoder.to(args.device)
    
    # Get transformer config
    transformer_config = {
        "max_text_seq_length": 512  # Default value from Wan pipeline
    }

    prompt_list = []
    data_list = []
    center_list = []
    scale_list = []
    
    for i in range(len(pexel_datalist_segment)):
        data = next(train_dataset_iterator)
        if data is None:
            continue
            
        # Get caption and prepare prompt
        caption = data.video.caption
        prompt_suffix = 'POINTMAP_STYLE.'
        full_prompt = caption + ' ' + prompt_suffix
        prompt_list.append(caption)
        
        # Generate video name
        if 'dynamic' in data.video.path:
            video_name = os.path.basename(data.video.path).replace('.mp4', '')
        else:
            video_name = data.video.path.split('/')[-2]
        if data.clip_start > -1:
            video_name = video_name + f'-{data.clip_start:04d}-{data.clip_start+data.length:04d}'
        
        # Define file paths
        video_path = data_dir / "videos" / f"{video_name}.mp4"
        pm_path = data_dir / "pointmap" / f"{video_name}.mp4"
        pointmap_latent_path = data_dir / "pointmap_latents" / f"{video_name}.pt"
        data_list.append(f'videos/{video_name}.mp4')
        
        # Cache prompt embedding
        prompt_hash = str(hashlib.sha256(full_prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        
        if not prompt_embedding_path.exists():
            prompt_embedding = encode_text(
                full_prompt, 
                text_tokenizer, 
                text_encoder, 
                transformer_config["max_text_seq_length"],
                args.device
            )
            prompt_embedding = prompt_embedding[0].to("cpu")
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            print(f"Saved prompt embedding to {prompt_embedding_path}")
        
        # Process RGB video if it doesn't exist
        if not video_path.exists():
            rgb = data.rgb_raw
            T, H, W, C = rgb.shape
            target_h, target_w = resolution_out
            
            # Calculate resize dimensions maintaining aspect ratio
            if H/W < target_h/target_w:
                new_h = target_h
                new_w = int(W * (target_h/H))
            else:
                new_w = target_w 
                new_h = int(H * (target_w/W))
                
            # Resize rgb to match shortest side
            rgb = nnf.interpolate(torch.from_numpy(rgb.transpose(0,3,1,2)), 
                                size=(new_h, new_w), 
                                mode='bilinear', 
                                align_corners=False,
                                antialias=True).numpy().transpose(0,2,3,1)
            rgb = F.center_crop(torch.from_numpy(rgb.transpose(0,3,1,2)), resolution_out).numpy().transpose(0,2,3,1)

            # Save first frame as image
            first_frame = rgb[0]
            first_frame_path = data_dir / "first_frames" / f"{video_name}.png"
            imageio.imwrite(first_frame_path, (first_frame * 255).clip(0, 255).astype(np.uint8))
            print(f"Saved first frame to {first_frame_path}")

            # Extend video length if needed
            if rgb.shape[0] < max_frames:
                num_original = T
                num_needed = max_frames - num_original
                reverse_rgb = rgb[::-1]
                rgb = np.concatenate([
                    rgb,
                    reverse_rgb[:num_needed]
                ], axis=0)
            
            # Save the processed RGB video
            imageio.mimwrite(video_path, (rgb * 255).clip(0, 255).astype(np.uint8), fps=24)
            print(f"Saved RGB video to {video_path}")
        else:
            print(f"RGB video already exists at {video_path}, skipping processing")
            rgb = None
        
        # Cache encoded video and image embeddings
        encoded_video_path = video_latent_dir / (video_name + ".safetensors")
        if not encoded_video_path.exists():
            if rgb is None:
                import decord
                decord.bridge.set_bridge("torch")
                video_reader = decord.VideoReader(uri=str(video_path))
                rgb = video_reader[:].float() / 255.0
                
                # Save first frame if it doesn't exist
                first_frame_path = data_dir / "first_frames" / f"{video_name}.png"
                if not first_frame_path.exists():
                    first_frame = rgb[0].cpu().numpy()
                    first_frame = first_frame.transpose(1, 2, 0)  # CHW to HWC
                    imageio.imwrite(first_frame_path, (first_frame * 255).clip(0, 255).astype(np.uint8))
                    print(f"Saved first frame to {first_frame_path}")
            
            # Prepare video tensor for encoding
            rgb_tensor = torch.from_numpy(rgb).float() if isinstance(rgb, np.ndarray) else rgb
            rgb_tensor = rgb_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
            rgb_tensor = rgb_tensor.unsqueeze(0)  # Add batch dimension
            
            # Encode video and first frame
            encoded_video = encode_video(rgb_tensor, vae)
            first_frame = (rgb_tensor[:, 0] + 1) * 0.5  # Get first frame
            # Convert first frame tensor to PIL image in RGB format
            first_frame = (first_frame[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            first_frame = PIL.Image.fromarray(first_frame, mode='RGB')
            image_embedding = encode_image(first_frame, image_processor, image_encoder, args.device)
            
            # Save encoded video and image embedding
            encoded_video = encoded_video[0].cpu()
            image_embedding = image_embedding[0].cpu()
            save_file({
                "encoded_video": encoded_video,
                "image_embedding": image_embedding
            }, encoded_video_path)
            print(f"Saved encoded video and image embedding to {encoded_video_path}")
        
        # Process pointmap if it doesn't exist
        if not pm_path.exists() or not pointmap_latent_path.exists():
            pointmap = data.pointmap.pcd
            pointmap = transform_pointclouds_to_first_camera(pointmap, data.pointmap.cams2world)
            pointmap = np.clip(pointmap, np.percentile(pointmap, 2, axis=1, keepdims=True), 
                               np.percentile(pointmap, 98, axis=1, keepdims=True))
            
            # Normalize pointmap
            pointmap_min = pointmap.min(axis=1, keepdims=True)
            pointmap_max = pointmap.max(axis=1, keepdims=True)
            center = (pointmap_min + pointmap_max) / 2
            scale = (pointmap_max - pointmap_min) / 2
            scale = scale.max() # same scale for all axes
            pointmap_in = (pointmap - center) / scale # this will normalize pointmap to [-1, 1]!!!
            
            center_list.append(center)
            scale_list.append(scale)
            
            # Reshape and process pointmap
            pointmap_in = pointmap_in.reshape(*data.pointmap.rgb.shape)
            
            T, H, W, C = data.rgb_raw.shape
            target_h, target_w = resolution_out
            
            if H/W < target_h/target_w:
                new_h = target_h
                new_w = int(W * (target_h/H))
            else:
                new_w = target_w 
                new_h = int(H * (target_w/W))
            
            pointmap_in = nnf.interpolate(torch.from_numpy(pointmap_in.transpose(0,3,1,2)), 
                                size=(new_h, new_w), 
                                mode='nearest').numpy().transpose(0,2,3,1)
            pointmap_in = F.center_crop(torch.from_numpy(pointmap_in.transpose(0,3,1,2)), 
                                        resolution_crop).numpy().transpose(0,2,3,1)
            
            if pointmap_in.shape[0] < max_frames:
                num_needed = max_frames - pointmap_in.shape[0]
                reverse_pointmap_in = pointmap_in[::-1]
                pointmap_in = np.concatenate([
                    pointmap_in,
                    reverse_pointmap_in[:num_needed]
                ], axis=0)
            
            # Save pointmap video and encode latents using Wan VAE
            imageio.mimwrite(pm_path, ((pointmap_in + 1) * 127.5).clip(0, 255).astype(np.uint8), fps=24)
            pointmap_tensor = torch.from_numpy(pointmap_in).float()
            pointmap_tensor = pointmap_tensor.unsqueeze(0).to(args.device)  # Add batch dimension
            pointmap_latents = encode_video(pointmap_tensor, vae)
            torch.save(pointmap_latents[0].cpu(), pointmap_latent_path)
            print(f"Saved pointmap video to {pm_path} and latents to {pointmap_latent_path}")
        else:
            print(f"Pointmap already exists at {pm_path}, skipping processing")
            if not center_list:
                try:
                    center_path = data_dir / "center" / f"{args.task_idx:05d}.npy"
                    scale_path = data_dir / "scale" / f"{args.task_idx:05d}.npy"
                    if center_path.exists() and scale_path.exists():
                        centers = np.load(center_path)
                        scales = np.load(scale_path)
                        center_list = centers.tolist()
                        scale_list = scales.tolist()
                except Exception as e:
                    print(f"Could not load center and scale: {e}")
        
        print(f'Finished processing sample {i+1}/{len(pexel_datalist_segment)}.')

    # Save metadata
    with open(data_dir / "prompts.txt", 'w') as f:
        f.write('\n'.join(prompt_list))
    
    with open(data_dir / "videos.txt", 'w') as f:
        f.write('\n'.join(data_list))

    # if center_list and scale_list:
    #     center_path = data_dir / "center" / f"{args.task_idx:05d}.npy"
    #     scale_path = data_dir / "scale" / f"{args.task_idx:05d}.npy"
    #     np.save(center_path, np.stack(center_list))
    #     np.save(scale_path, np.stack(scale_list))

    print(f'All finished.')

def generate_datalist_from_directory(data_dir):
    """
    Generate datalist from directory structure.
    For dynamic: traverse to the deepest folder
    For static: find all .npz files
    """
    data_dir = Path(data_dir).absolute()
    datalist = []
    
    # Process dynamic data
    dynamic_dir = data_dir / "dynamic"
    if dynamic_dir.exists():
        for root, dirs, files in os.walk(dynamic_dir):
            # If this is the deepest level (no subdirectories) or contains video files
            if not dirs or any(file.endswith('.mp4') for file in files):
                datalist.append(str(Path(root).absolute()) + '/')
            for file in files:
                if file.endswith('.npz'):
                    full_path = Path(root) / file
                    datalist.append(str(full_path.absolute()))
    
    # Process static data
    static_dir = data_dir / "static"
    if static_dir.exists():
        for root, dirs, files in os.walk(static_dir):
            for file in files:
                if file.endswith('.npz'):
                    full_path = Path(root) / file
                    datalist.append(str(full_path.absolute()))
    
    return datalist

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build wan dataset')
    parser.add_argument('--data_dir', type=str, default='./data', help='Root directory containing dynamic and static folders')
    parser.add_argument('--out', type=str, default='./data/wan21')
    parser.add_argument('--model_path', type=str, default='./pretrained/Wan2.1-I2V-14B-480P-Diffusers')
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--resolution_h', type=int, default=480)
    parser.add_argument('--resolution_w', type=int, default=720)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)
