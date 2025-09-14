import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
import PIL

from core.finetune.constants import LOG_LEVEL, LOG_NAME

from .utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
    generate_uniform_pointmap,
)


if TYPE_CHECKING:
    from core.finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseWanDataset(Dataset):
    """
    Base dataset class for Image-to-Video (I2V) training.

    This dataset loads prompts, videos and corresponding conditioning images for I2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        image_column (str): Path to file containing image paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        image_column: str | None,
        device: torch.device,
        trainer: "Trainer" = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        data_root = Path(data_root)
        self.prompts = load_prompts(data_root / caption_column)
        self.videos = load_videos(data_root / video_column)
        if image_column is not None:
            self.images = load_images(data_root / image_column)
        else:
            self.images = load_images_from_videos(self.videos)
        self.trainer = trainer

        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text

        uniform_pointmap = torch.from_numpy(generate_uniform_pointmap(self.trainer.args.train_resolution[1], self.trainer.args.train_resolution[2])).permute(2, 0, 1)
        self.uniform_pointmap = uniform_pointmap * 2 - 1

        # Check if number of prompts matches number of videos and images
        if not (len(self.videos) == len(self.prompts) == len(self.images)):
            raise ValueError(
                f"Expected length of prompts, videos and images to be the same but found {len(self.prompts)}, {len(self.videos)} and {len(self.images)}. Please ensure that the number of caption prompts, videos and images match in your dataset."
            )

        # Check if all video files exist
        if any(not path.is_file() for path in self.videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.videos if not path.is_file())}"
            )

        # Check if all image files exist
        if any(not path.is_file() for path in self.images):
            raise ValueError(
                f"Some image files were not found. Please ensure that all image files exist in the dataset directory. Missing file: {next(path for path in self.images if not path.is_file())}"
            )

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        while True:
            try:
                ret = self.getitem(index)
                if ret['video_metadata']['num_frames'] != 13:
                    raise ValueError("Not enough frames")
                break
            except Exception as e:
                # print(e)
                index = (index + 1) % len(self.videos)
        return ret

    def getitem(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            return index

        prompt = self.prompts[index]
        # HACK: add suffix prompt
        suffix = 'POINTMAP_STYLE.'
        prompt = prompt + ' ' + suffix
        video = self.videos[index]
        image = self.images[index]
        # Use the correct resolution string for WAN dataset
        train_resolution = self.trainer.args.train_resolution
        if isinstance(train_resolution, (list, tuple)):
            train_resolution_str = f"{train_resolution[0]}x{train_resolution[1]}x{train_resolution[2]}"
        else:
            train_resolution_str = str(train_resolution)
        cache_dir = self.trainer.args.data_root / "cache"
        video_latent_dir = cache_dir / "video_latent" / "wan-i2v" / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        video_name = video.stem
        encoded_video_path = video_latent_dir / (video_name + ".safetensors")
        pm_latent_dir = self.trainer.args.data_root / "pointmap_latents"
        encoded_pm_path = pm_latent_dir / (video_name + ".pt")

        # Load prompt embedding
        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            prompt_embedding = prompt_embedding[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)

        # Load encoded video and image embedding
        if encoded_video_path.exists():
            loaded = load_file(encoded_video_path)
            encoded_video = loaded["encoded_video"]
            image_embedding = loaded["image_embedding"]
            logger.debug(f"Loaded encoded video and image embedding from {encoded_video_path}", main_process_only=False)
        else:
            raise FileNotFoundError(f"Encoded video file not found: {encoded_video_path}")

        # Load encoded pointmap
        if encoded_pm_path.exists():
            encoded_pm = torch.load(encoded_pm_path, map_location='cpu')
            logger.debug(f"Loaded encoded point map from {encoded_pm_path}", main_process_only=False)
        else:
            raise FileNotFoundError(f"Encoded pointmap file not found: {encoded_pm_path}")

        # Load first frame image
        _, image = self.preprocess(None, self.images[index])    # resize image
        image = self.image_transform(image)

        encoded_pm_mean = -0.13
        encoded_pm_std = 1.70
        encoded_pm = (encoded_pm - encoded_pm_mean) / encoded_pm_std
        # HACK: train on the first 49 frames
        encoded_video = torch.concat([encoded_video[:, :13, :, :], encoded_pm[:, :13, :, :]], -1)
        image = torch.concat([image, self.uniform_pointmap], -1)

        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "image_embedding": image_embedding,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }

    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses a video and an image.
        If either path is None, no preprocessing will be done for that input.

        Args:
            video_path: Path to the video file to load
            image_path: Path to the image file to load

        Returns:
            A tuple containing:
                - video(torch.Tensor) of shape [F, C, H, W] where F is number of frames,
                  C is number of channels, H is height and W is width
                - image(torch.Tensor) of shape [C, H, W]
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to an image.

        Args:
            image (torch.Tensor): A 3D tensor representing an image
                with shape [C, H, W] where:
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed image tensor
        """
        raise NotImplementedError("Subclass must implement this method")


class WanI2VDatasetWithResize(BaseWanDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width)
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)