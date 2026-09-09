"""Appearance Fidelity metrics.

Low-level visual preservation between paired frames: PSNR, SSIM (color,
texture, structural layout) and LPIPS (perceptual appearance). Most sensitive
to surface-level repainting and local visual drift.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as _compute_psnr
from skimage.metrics import structural_similarity as _compute_ssim
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode


def compute_psnr_ssim(img_a: np.ndarray, img_b: np.ndarray) -> dict:
    """Compute PSNR and SSIM between two RGB uint8 numpy arrays.

    Returns a dict with rounded 'psnr' and 'ssim'.
    """
    psnr = float(_compute_psnr(img_a, img_b, data_range=255))
    ssim = float(_compute_ssim(img_a, img_b, data_range=255, channel_axis=2, win_size=7))
    return {"psnr": round(psnr, 4), "ssim": round(ssim, 6)}


class LpipsClipComputer:
    """Compute LPIPS distance and CLIP-Video score."""

    def __init__(self, clip_checkpoint_path=None, device="cuda:0"):
        import lpips as lpips_lib
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # LPIPS (AlexNet)
        self.lpips_fn = lpips_lib.LPIPS(net="alex").to(self.device)
        self.lpips_fn.eval()

        # CLIP (TorchScript checkpoint)
        if clip_checkpoint_path is None:
            clip_checkpoint_path = os.path.expanduser("~/.cache/clip/ViT-B-32.pt")
        self.clip_device = self.device
        # Load to CPU first, then move to target device to avoid mixed-device issues
        self.clip_model = torch.jit.load(clip_checkpoint_path, map_location="cpu")
        self.clip_model = self.clip_model.to(self.clip_device)
        self.clip_model.eval()
        self.clip_preprocess = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(224),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                      std=(0.26862954, 0.26130258, 0.27577711)),
        ])
        print(f"  LPIPS (AlexNet) + CLIP (ViT-B-32) loaded from {clip_checkpoint_path}")

    def compute_lpips(self, img_a_rgb, img_b_rgb):
        """Compute LPIPS distance between two RGB uint8 numpy arrays."""
        tensor_a = torch.from_numpy(img_a_rgb).float().permute(2, 0, 1) / 255.0
        tensor_a = (tensor_a * 2.0 - 1.0).unsqueeze(0).to(self.device)
        tensor_b = torch.from_numpy(img_b_rgb).float().permute(2, 0, 1) / 255.0
        tensor_b = (tensor_b * 2.0 - 1.0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            distance = self.lpips_fn(tensor_a, tensor_b)
        return distance.item()

    def compute_clip_video_score(self, frame_list):
        """Compute CLIP-Video: mean cosine similarity of consecutive frames.

        Args:
            frame_list: list of RGB uint8 numpy arrays, sorted by frame index.

        Returns:
            float: mean cosine similarity across consecutive pairs.
        """
        if len(frame_list) < 2:
            return 1.0

        features = []
        for frame_rgb in frame_list:
            tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
            preprocessed = self.clip_preprocess(tensor).unsqueeze(0).to(self.clip_device).half()
            with torch.no_grad():
                feat = self.clip_model.encode_image(preprocessed).float()
            features.append(feat)

        features = torch.cat(features, dim=0)
        features = F.normalize(features, dim=1)

        similarities = []
        for i in range(len(features) - 1):
            sim = (features[i] * features[i + 1]).sum().item()
            similarities.append(sim)

        return float(np.mean(similarities))
