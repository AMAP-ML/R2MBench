"""Scene Identity Preservation metrics.

Whether paired frames still depict the same place at the scene level. Uses
DINOv2 as a general visual representation for semantic scene similarity, plus
two visual-place-recognition descriptors (BoQ, MutualVPR) built for identifying
previously visited locations. For all descriptors we extract global features,
L2-normalize, and compute cosine similarity between frame pairs.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"{name} is not configured. Please set it in run_config.conf.")
    return value

def _prepend_path_from_env(name: str) -> None:
    repo_path = _required_env(name)
    import_path = os.path.join(repo_path, "src")
    if not os.path.isdir(import_path):
        import_path = repo_path
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

TORCH_HOME = _required_env("TORCH_HOME")
TORCH_HUB_DIR = os.path.join(TORCH_HOME, "hub")
DINOV2_CHECKPOINT = os.path.join(TORCH_HUB_DIR, "checkpoints/dinov2_vitb14_pretrain.pth")
BOQ_CHECKPOINT = _required_env("BOQ_CHECKPOINT")
MUTUALVPR_CHECKPOINT = _required_env("MUTUALVPR_CHECKPOINT")

class DINOv2FeatureExtractor:
    """Extract features from images using DINOv2 (loaded from local cache)."""

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda:0"):
        self.device = device
        print(f"  Loading DINOv2 model: {model_name} on {device}...")
        print(f"  Using local hub dir: {TORCH_HUB_DIR}")

        local_repo_path = os.path.join(TORCH_HUB_DIR, "facebookresearch_dinov2_main")
        self.model = torch.hub.load(
            local_repo_path,
            model_name,
            source="local",
            pretrained=False,
        )
        state_dict = torch.load(DINOV2_CHECKPOINT, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"  DINOv2 model loaded successfully.")

    @torch.no_grad()
    def extract_feature(self, image: np.ndarray) -> torch.Tensor:
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        feature = self.model(tensor)
        return F.normalize(feature, dim=-1)

    @torch.no_grad()
    def similarity(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        feat_a = self.extract_feature(img_a)
        feat_b = self.extract_feature(img_b)
        return float(F.cosine_similarity(feat_a, feat_b, dim=-1).item())


class DinoV2Backbone(nn.Module):
    """DINOv2 backbone used by the BoQ scene-identity descriptor."""

    def __init__(self, backbone_name: str = "dinov2_vitb14"):
        super().__init__()
        self.backbone_name = backbone_name

        local_repo_path = os.path.join(TORCH_HUB_DIR, "facebookresearch_dinov2_main")
        self.dino = torch.hub.load(
            local_repo_path,
            backbone_name,
            source="local",
            pretrained=False,
        )
        state_dict = torch.load(DINOV2_CHECKPOINT, map_location="cpu", weights_only=True)
        self.dino.load_state_dict(state_dict, strict=True)

        for parameter in self.dino.parameters():
            parameter.requires_grad = False

        self.out_channels = self.dino.embed_dim

    @property
    def patch_size(self) -> int:
        return self.dino.patch_embed.patch_size[0]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = tensor.shape
        with torch.no_grad():
            tokens = self.dino.prepare_tokens_with_masks(tensor)
            for block in self.dino.blocks:
                tokens = block(tokens)

        tokens = tokens[:, 1:]
        _, _, channels = tokens.shape
        patch_size = self.patch_size
        return tokens.permute(0, 2, 1).view(
            batch_size,
            channels,
            height // patch_size,
            width // patch_size,
        )

class BoQVPRModel(nn.Module):
    """BoQ VPR model: DINOv2 backbone + Bag-of-Queries aggregator."""

    def __init__(self, backbone: DinoV2Backbone, aggregator: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        features = self.backbone(tensor)
        descriptor, _ = self.aggregator(features)
        return descriptor

def load_boq_model(device: str = "cuda:0") -> BoQVPRModel:
    """Load BoQ scene-identity model from local checkpoints."""
    _prepend_path_from_env("BOQ_ROOT")
    from boq import BoQ

    print("  Loading DINOv2 backbone from local cache...")
    backbone = DinoV2Backbone(backbone_name="dinov2_vitb14")

    print(f"  Creating BoQ aggregator (in_channels={backbone.out_channels}, output_dim=12288)...")
    aggregator = BoQ(
        in_channels=backbone.out_channels,
        proj_channels=384,
        num_queries=64,
        num_layers=2,
        row_dim=12288 // 384,
    )

    model = BoQVPRModel(backbone=backbone, aggregator=aggregator)

    print(f"  Loading BoQ weights from: {BOQ_CHECKPOINT}")
    state_dict = torch.load(BOQ_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    print(f"  BoQ model loaded on {device}.")
    return model

class BoQFeatureExtractor:
    """Extract BoQ descriptors from images using DinoV2Backbone + BoQVPRModel."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        print(f"  Loading BoQ model on {device}...")

        self.model = load_boq_model(device=device)

        self.transform = transforms.Compose([
            transforms.Resize((322, 322), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"  BoQ model loaded successfully.")

    @torch.no_grad()
    def extract_feature(self, image: np.ndarray) -> torch.Tensor:
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        descriptor = self.model(tensor)
        return F.normalize(descriptor, dim=-1)

    @torch.no_grad()
    def similarity(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        feat_a = self.extract_feature(img_a)
        feat_b = self.extract_feature(img_b)
        return float(F.cosine_similarity(feat_a, feat_b, dim=-1).item())


class MutualVPRFeatureExtractor:
    """Extract MutualVPR descriptors (DINOv2 + GeM + Linear, 512-dim)."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        print(f"  Loading MutualVPR model on {device}...")

        _prepend_path_from_env("MUTUALVPR_ROOT")
        from cosplace_model.cosplace_network import MutualVPR

        self.model = MutualVPR(pretrained_foundation=False, output_dim=512)

        state_dict = torch.load(MUTUALVPR_CHECKPOINT, map_location="cpu")
        # Remove 'module.' prefix from DataParallel checkpoint
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(cleaned, strict=False)
        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"  MutualVPR model loaded successfully.")

    @torch.no_grad()
    def extract_feature(self, image: np.ndarray) -> torch.Tensor:
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        descriptor = self.model(tensor)
        return F.normalize(descriptor, dim=-1)

    @torch.no_grad()
    def similarity(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        feat_a = self.extract_feature(img_a)
        feat_b = self.extract_feature(img_b)
        return float(F.cosine_similarity(feat_a, feat_b, dim=-1).item())
