"""Object Identity metrics.

Whether salient objects in the earlier frame of a pair persist as the same
objects in the later frame. Object anchors are mined with GroundingDINO and
refined into masks with SAM2. In the paired frame each anchor concept is
re-detected independently (rather than tracked through the intervening video),
so tracker drift is not mistaken for memory. For each accepted match, appearance
similarity comes from cosine similarity of masked DINOv2 crop features, and
semantic consistency combines CLIP image-text similarity with GroundingDINO
confidence. We report appearance persistence and semantic persistence, where
unmatched anchors contribute zero, so the score reflects both object identity
quality and object survival.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu as decord_cpu
from PIL import Image

try:
    from .scene_identity import DINOv2FeatureExtractor
except ImportError:
    metrics_dir = Path(__file__).resolve().parent
    if str(metrics_dir) not in sys.path:
        sys.path.insert(0, str(metrics_dir))
    from scene_identity import DINOv2FeatureExtractor

DEFAULT_TEXT_PROMPT = (
    "object . furniture . building . vehicle . plant . person . animal . sign . "
    "door . window . chair . table . sofa . bed . cabinet . lamp . screen . "
    "picture . shelf . tree . road . wall . floor ."
)

def _prepend_path_from_env(name: str) -> None:
    repo_path = os.environ.get(name)
    if not repo_path:
        raise EnvironmentError(f"{name} is not configured. Please set it in run_config.conf.")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

@dataclass
class ObjectInstance:
    bbox_xyxy: List[float]
    phrase: str
    concept: str
    box_score: float
    mask_score: float
    area_ratio: float
    feature: torch.Tensor
    clip_feature: Optional[torch.Tensor] = None

def _load_groundingdino(config_path: str, checkpoint_path: str, device: str):
    _prepend_path_from_env("GROUNDINGDINO_ROOT")
    try:
        from groundingdino.util.inference import load_model
    except ImportError as exc:
        raise ImportError(
            "GroundingDINO is not installed or not on PYTHONPATH. "
            "Install it or run with its repository on PYTHONPATH."
        ) from exc
    return load_model(config_path, checkpoint_path, device=device)


def _groundingdino_transform(image_np: np.ndarray):
    _prepend_path_from_env("GROUNDINGDINO_ROOT")
    try:
        import groundingdino.datasets.transforms as T
    except ImportError as exc:
        raise ImportError(
            "Cannot import groundingdino.datasets.transforms. "
            "Check your GroundingDINO installation."
        ) from exc

    image_pil = Image.fromarray(image_np).convert("RGB")
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)
    return image_tensor


def _load_sam2(config_path: str, checkpoint_path: str, device: str):
    _prepend_path_from_env("SAM2_ROOT")
    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import _load_checkpoint
    except ImportError as exc:
        raise ImportError(
            "SAM2 is not installed or not on PYTHONPATH. "
            "Install SAM2 or run with its repository on PYTHONPATH."
        ) from exc

    # Load config directly from absolute path (bypass Hydra's compose which
    # only accepts relative config names within its search path).
    cfg = OmegaConf.load(config_path)
    # Apply the same postprocessing overrides that build_sam2 uses by default
    OmegaConf.update(cfg, "model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability", True, merge=True)
    OmegaConf.update(cfg, "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta", 0.05, merge=True)
    OmegaConf.update(cfg, "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh", 0.98, merge=True)
    OmegaConf.resolve(cfg)

    sam2_model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(sam2_model, checkpoint_path)
    sam2_model = sam2_model.to(device)
    sam2_model.eval()
    return SAM2ImagePredictor(sam2_model)


def _predict_sam2_masks(predictor, image_np: np.ndarray, boxes_xyxy: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Predict one SAM2 mask per input box.

    SAM2's image predictor API is box-conditioned and returns numpy masks.
    We call it per box to avoid depending on optional batch helper variants.
    """
    predictor.set_image(image_np)
    masks_out = []
    scores_out = []
    for box in boxes_xyxy.detach().cpu().numpy().astype(np.float32):
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )
        if masks.ndim == 3:
            masks_out.append(masks[0].astype(bool))
        else:
            masks_out.append(masks.astype(bool))
        if np.ndim(scores) > 0:
            scores_out.append(float(scores[0]))
        else:
            scores_out.append(float(scores))
    if not masks_out:
        return np.zeros((0, image_np.shape[0], image_np.shape[1]), dtype=bool), np.zeros((0,), dtype=np.float32)
    return np.stack(masks_out, axis=0), np.asarray(scores_out, dtype=np.float32)


def _cxcywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    cx, cy, bw, bh = boxes.unbind(-1)
    x0 = (cx - 0.5 * bw) * width
    y0 = (cy - 0.5 * bh) * height
    x1 = (cx + 0.5 * bw) * width
    y1 = (cy + 0.5 * bh) * height
    xyxy = torch.stack([x0, y0, x1, y1], dim=-1)
    xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, width - 1)
    xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, height - 1)
    return xyxy


def _nms_boxes(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)
    try:
        from torchvision.ops import nms

        return nms(boxes_xyxy, scores, iou_threshold)
    except Exception:
        order = torch.argsort(scores, descending=True)
        return order


def _clean_concept(text: str) -> str:
    """Normalize a detector phrase into a compact object concept prompt."""
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9 /_-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text or "object"


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


class CLIPFeatureExtractor:
    """Extract CLIP image/text features for semantic scoring."""

    def __init__(self, clip_checkpoint: str, device: str = "cuda:0"):
        import clip as clip_lib
        from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode, ToTensor

        self.device = device
        print(f"  Loading CLIP (ViT-B-32) via openai/clip from {clip_checkpoint}...")
        # Load via the clip library which provides both image and text encoders
        self.model, self.preprocess = clip_lib.load(clip_checkpoint, device=device)
        self.model.eval()
        self.clip_tokenize = clip_lib.tokenize
        print(f"  CLIP model loaded successfully.")

    @torch.no_grad()
    def extract_image_feature(self, image: np.ndarray) -> torch.Tensor:
        pil_image = Image.fromarray(image)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        feature = self.model.encode_image(tensor).float()
        return F.normalize(feature, dim=-1)

    @torch.no_grad()
    def extract_text_feature(self, text: str) -> torch.Tensor:
        tokens = self.clip_tokenize([text]).to(self.device)
        feature = self.model.encode_text(tokens).float()
        return F.normalize(feature, dim=-1)


class GroundedSamObjectEvaluator:
    """Mine first-frame objects, then re-detect each concept in the paired frame."""

    def __init__(
        self,
        groundingdino_config: str,
        groundingdino_checkpoint: str,
        sam_checkpoint: str,
        sam2_config: str,
        dino_extractor: DINOv2FeatureExtractor,
        clip_extractor: Optional[CLIPFeatureExtractor] = None,
        device: str = "cuda:0",
        text_prompt: str = DEFAULT_TEXT_PROMPT,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        nms_iou: float = 0.60,
        max_objects: int = 20,
        min_area_ratio: float = 0.002,
        match_threshold: float = 0.50,
        max_candidates_per_concept: int = 5,
        appearance_weight: float = 0.7,
        semantic_weight: float = 0.3,
    ):
        self.device = device
        self.text_prompt = text_prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou = nms_iou
        self.max_objects = max_objects
        self.min_area_ratio = min_area_ratio
        self.match_threshold = match_threshold
        self.max_candidates_per_concept = max_candidates_per_concept
        self.appearance_weight = appearance_weight
        self.semantic_weight = semantic_weight
        self.dino = dino_extractor
        self.clip = clip_extractor
        self.grounding_model = _load_groundingdino(
            groundingdino_config, groundingdino_checkpoint, device
        )
        self.sam_predictor = _load_sam2(sam2_config, sam_checkpoint, device)
        self.frame_cache: Dict[Tuple[int, str, int], List[ObjectInstance]] = {}

    @torch.no_grad()
    def detect_objects(
        self,
        image_np: np.ndarray,
        frame_id: Optional[int] = None,
        text_prompt: Optional[str] = None,
        max_objects: Optional[int] = None,
    ) -> List[ObjectInstance]:
        prompt = text_prompt if text_prompt is not None else self.text_prompt
        max_keep = self.max_objects if max_objects is None else max_objects
        cache_key = None
        if frame_id is not None:
            cache_key = (int(frame_id), prompt, int(max_keep))
            if cache_key in self.frame_cache:
                return self.frame_cache[cache_key]

        try:
            from groundingdino.util.inference import predict
        except ImportError as exc:
            raise ImportError("Cannot import GroundingDINO predict function.") from exc

        height, width = image_np.shape[:2]
        image_tensor = _groundingdino_transform(image_np)
        boxes, logits, phrases = predict(
            model=self.grounding_model,
            image=image_tensor,
            caption=prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        if boxes is None or len(boxes) == 0:
            objects: List[ObjectInstance] = []
            if cache_key is not None:
                self.frame_cache[cache_key] = objects
            return objects

        boxes = boxes.to(self.device)
        logits = logits.to(self.device)
        boxes_xyxy = _cxcywh_to_xyxy(boxes, width, height)

        keep = _nms_boxes(boxes_xyxy, logits, self.nms_iou)
        if max_keep > 0:
            keep = keep[:max_keep]
        boxes_xyxy = boxes_xyxy[keep]
        logits = logits[keep]
        phrases = [phrases[int(i)] for i in keep.detach().cpu().tolist()]

        if len(boxes_xyxy) == 0:
            objects = []
            if cache_key is not None:
                self.frame_cache[cache_key] = objects
            return objects

        masks, mask_scores = _predict_sam2_masks(self.sam_predictor, image_np, boxes_xyxy)

        objects = []
        image_area = float(max(width * height, 1))
        for idx, mask in enumerate(masks):
            area_ratio = float(mask.sum() / image_area)
            if area_ratio < self.min_area_ratio:
                continue

            x0, y0, x1, y1 = [int(round(v)) for v in boxes_xyxy[idx].detach().cpu().tolist()]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(width - 1, x1), min(height - 1, y1)
            if x1 <= x0 or y1 <= y0:
                continue

            crop = image_np[y0 : y1 + 1, x0 : x1 + 1].copy()
            crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
            if crop_mask.any():
                crop[~crop_mask] = 255

            phrase = str(phrases[idx])
            dino_feature = self.dino.extract_feature(crop).detach().cpu()
            clip_feature = self.clip.extract_image_feature(crop).detach().cpu() if self.clip is not None else None
            objects.append(
                ObjectInstance(
                    bbox_xyxy=[float(x0), float(y0), float(x1), float(y1)],
                    phrase=phrase,
                    concept=_clean_concept(phrase),
                    box_score=float(logits[idx].detach().cpu().item()),
                    mask_score=float(mask_scores[idx]),
                    area_ratio=area_ratio,
                    feature=dino_feature,
                    clip_feature=clip_feature,
                )
            )

        objects.sort(key=lambda item: item.box_score * max(item.area_ratio, 1e-6), reverse=True)
        if max_keep > 0:
            objects = objects[:max_keep]
        if cache_key is not None:
            self.frame_cache[cache_key] = objects
        return objects

    def score_pair(self, image_a: np.ndarray, image_b: np.ndarray, frame_a: int, frame_b: int) -> dict:
        anchors = self.detect_objects(
            image_a,
            frame_id=frame_a,
            text_prompt=self.text_prompt,
            max_objects=self.max_objects,
        )
        num_anchors = len(anchors)

        result = {
            "num_anchor_objects": num_anchors,
            "num_detected_candidates": 0,
            "num_object_matches": 0,
            "object_existence_rate": 0.0,
            "object_appearance_similarity": 0.0,
            "object_semantic_similarity": 0.0,
            "object_appearance_persistence": 0.0,
            "object_semantic_persistence": 0.0,
            "object_recall": 0.0,
            "object_identity_similarity": 0.0,
            "object_consistency": 0.0,
            "anchor_objects": [],
            "object_matches": [],
        }
        if num_anchors == 0:
            return result

        object_scores = []
        matched_app_scores = []
        matched_sem_scores = []
        object_matches = []
        anchor_objects = []
        total_candidates = 0
        weight_sum = max(self.appearance_weight + self.semantic_weight, 1e-8)

        for anchor_idx, anchor in enumerate(anchors):
            concept = anchor.concept or _clean_concept(anchor.phrase)
            candidates = self.detect_objects(
                image_b,
                frame_id=frame_b,
                text_prompt=f"{concept} .",
                max_objects=self.max_candidates_per_concept,
            )
            total_candidates += len(candidates)
            anchor_objects.append(
                {
                    "idx": anchor_idx,
                    "concept": concept,
                    "phrase": anchor.phrase,
                    "box_score": round(float(anchor.box_score), 6),
                    "area_ratio": round(float(anchor.area_ratio), 6),
                    "bbox_xyxy": anchor.bbox_xyxy,
                }
            )

            best = None
            best_score = -float("inf")
            for cand_idx, candidate in enumerate(candidates):
                raw_app = float(F.cosine_similarity(anchor.feature, candidate.feature, dim=-1).item())
                app_score = _clip01((raw_app + 1.0) * 0.5)
                # Semantic score: does the candidate truly match the anchor's
                # text concept? Combines GroundingDINO box_score (grounding
                # confidence) with CLIP image-text similarity when available.
                grounding_conf = _clip01(candidate.box_score)
                if self.clip is not None and candidate.clip_feature is not None:
                    concept_text = anchor.concept or _clean_concept(anchor.phrase)
                    text_feat = self.clip.extract_text_feature(concept_text)
                    clip_sim = _clip01((float(F.cosine_similarity(
                        candidate.clip_feature.to(text_feat.device), text_feat, dim=-1
                    ).item()) + 1.0) * 0.5)
                    sem_score = 0.5 * grounding_conf + 0.5 * clip_sim
                else:
                    sem_score = grounding_conf
                match_score = (
                    self.appearance_weight * app_score
                    + self.semantic_weight * sem_score
                ) / weight_sum
                if match_score > best_score:
                    best_score = match_score
                    best = {
                        "candidate_idx": cand_idx,
                        "candidate": candidate,
                        "appearance": app_score,
                        "semantic": sem_score,
                        "score": match_score,
                    }

            if best is not None and best["score"] >= self.match_threshold:
                candidate = best["candidate"]
                object_scores.append(best["score"])
                matched_app_scores.append(best["appearance"])
                matched_sem_scores.append(best["semantic"])
                object_matches.append(
                    {
                        "anchor_idx": anchor_idx,
                        "candidate_idx": best["candidate_idx"],
                        "concept": concept,
                        "score": round(float(best["score"]), 6),
                        "appearance": round(float(best["appearance"]), 6),
                        "semantic": round(float(best["semantic"]), 6),
                        "anchor_phrase": anchor.phrase,
                        "candidate_phrase": candidate.phrase,
                        "candidate_box_score": round(float(candidate.box_score), 6),
                        "candidate_bbox_xyxy": candidate.bbox_xyxy,
                    }
                )
            else:
                object_scores.append(0.0)

        num_matches = len(object_matches)
        existence = num_matches / max(num_anchors, 1)
        appearance = float(np.mean(matched_app_scores)) if matched_app_scores else 0.0
        semantic = float(np.mean(matched_sem_scores)) if matched_sem_scores else 0.0

        # Separate persistence: each uses its own score with 0 for unmatched anchors
        app_persistence_scores = []
        sem_persistence_scores = []
        match_idx = 0
        for anchor_idx in range(num_anchors):
            if match_idx < num_matches and object_matches[match_idx]["anchor_idx"] == anchor_idx:
                app_persistence_scores.append(object_matches[match_idx]["appearance"])
                sem_persistence_scores.append(object_matches[match_idx]["semantic"])
                match_idx += 1
            else:
                app_persistence_scores.append(0.0)
                sem_persistence_scores.append(0.0)
        app_persistence = float(np.mean(app_persistence_scores)) if app_persistence_scores else 0.0
        sem_persistence = float(np.mean(sem_persistence_scores)) if sem_persistence_scores else 0.0

        result.update(
            {
                "num_detected_candidates": int(total_candidates),
                "num_object_matches": int(num_matches),
                "object_existence_rate": round(float(existence), 6),
                "object_appearance_similarity": round(float(appearance), 6),
                "object_semantic_similarity": round(float(semantic), 6),
                "object_appearance_persistence": round(float(app_persistence), 6),
                "object_semantic_persistence": round(float(sem_persistence), 6),
                "object_recall": round(float(existence), 6),
                "object_identity_similarity": round(float(appearance), 6),
                "object_consistency": round(float(app_persistence), 6),
                "anchor_objects": anchor_objects,
                "object_matches": object_matches,
            }
        )
        return result

def evaluate_object_pairs(
    video_reader,
    pairs: Sequence[Tuple[int, int]],
    evaluator: GroundedSamObjectEvaluator,
    label: str,
) -> List[dict]:
    """Evaluate object identity for frame pairs."""
    results = []
    for pair_idx, (frame_a, frame_b) in enumerate(pairs):
        image_a = video_reader[frame_a].asnumpy()
        image_b = video_reader[frame_b].asnumpy()
        score = evaluator.score_pair(image_a, image_b, frame_a, frame_b)
        score.update({"status": "ok", "video_frames": [int(frame_a), int(frame_b)]})
        results.append(score)

        if (pair_idx + 1) % 10 == 0:
            print(f"      {label}: {pair_idx + 1}/{len(pairs)} done", flush=True)

    return results

def aggregate_object_metrics(results: Sequence[dict]) -> dict:
    """Aggregate object-identity pair scores."""
    metric_keys = [
        "object_existence_rate",
        "object_appearance_similarity",
        "object_semantic_similarity",
        "object_appearance_persistence",
        "object_semantic_persistence",
        "object_recall",
        "object_identity_similarity",
        "object_consistency",
        "num_anchor_objects",
        "num_detected_candidates",
        "num_object_matches",
    ]

    aggregate = {}
    for key in metric_keys:
        values = [
            float(item[key])
            for item in results
            if item.get("status") == "ok" and key in item
        ]
        if values:
            aggregate[key] = {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6),
                "median": round(float(np.median(values)), 6),
                "count": len(values),
            }

    return aggregate

def compute_object_nmr(
    revisit_metrics: dict,
    baseline_metrics: dict,
    short_metrics: dict,
    eps: float = 1e-8,
) -> dict:
    """Compute NMR values for object-identity metrics."""
    nmr = {}
    for key in [
        "object_existence_rate",
        "object_appearance_similarity",
        "object_semantic_similarity",
        "object_appearance_persistence",
        "object_semantic_persistence",
    ]:
        if key not in revisit_metrics or key not in baseline_metrics or key not in short_metrics:
            continue

        revisit_value = revisit_metrics[key]["mean"]
        baseline_value = baseline_metrics[key]["mean"]
        short_value = short_metrics[key]["mean"]
        dynamic_range = short_value - baseline_value
        memory_gain = revisit_value - baseline_value

        if dynamic_range <= eps:
            nmr[key] = {
                "value": None,
                "valid": False,
                "mg": round(float(memory_gain), 6),
                "dr": round(float(dynamic_range), 6),
            }
        else:
            nmr[key] = {
                "value": round(float(memory_gain / (dynamic_range + eps)), 6),
                "valid": True,
                "mg": round(float(memory_gain), 6),
                "dr": round(float(dynamic_range), 6),
            }

    return nmr

def save_object_results(output_dir: Path, per_video_results: dict) -> None:
    """Write partial object-family results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results_partial.json", "w") as output_file:
        json.dump({"per_video": per_video_results}, output_file, indent=2, ensure_ascii=False)

def build_object_global_summary(per_video_results: dict) -> dict:
    """Build global object-family summary across evaluated videos."""
    metric_keys = [
        "object_existence_rate",
        "object_appearance_similarity",
        "object_semantic_similarity",
        "object_appearance_persistence",
        "object_semantic_persistence",
        "object_recall",
        "object_identity_similarity",
        "object_consistency",
        "num_anchor_objects",
        "num_detected_candidates",
        "num_object_matches",
    ]
    split_values = {
        "revisit": {key: [] for key in metric_keys},
        "baseline": {key: [] for key in metric_keys},
        "short": {key: [] for key in metric_keys},
    }

    for video_result in per_video_results.values():
        if not isinstance(video_result, dict):
            continue
        for split_name, metrics in split_values.items():
            split_block = video_result.get(split_name, {})
            for key in metric_keys:
                if key in split_block and isinstance(split_block[key], dict):
                    metrics[key].append(float(split_block[key]["mean"]))

    summary = {}
    for split_name, metrics in split_values.items():
        summary[split_name] = {}
        for key, values in metrics.items():
            if values:
                summary[split_name][key] = round(float(np.mean(values)), 6)

    summary["nmr"] = {}
    for key in [
        "object_existence_rate",
        "object_appearance_similarity",
        "object_semantic_similarity",
        "object_appearance_persistence",
        "object_semantic_persistence",
    ]:
        if key not in summary["revisit"] or key not in summary["baseline"] or key not in summary["short"]:
            continue

        revisit_value = summary["revisit"][key]
        baseline_value = summary["baseline"][key]
        short_value = summary["short"][key]
        dynamic_range = short_value - baseline_value
        memory_gain = revisit_value - baseline_value
        summary["nmr"][key] = None if dynamic_range <= 1e-8 else round(
            float(memory_gain / (dynamic_range + 1e-8)),
            6,
        )

    return summary

def run_object_identity_cli() -> None:
    """CLI entry for the object-identity metric family."""
    full_eval_dir = Path(__file__).resolve().parents[1]
    if str(full_eval_dir) not in sys.path:
        sys.path.insert(0, str(full_eval_dir))

    from eval_revisit import (
        extract_revisit_pairs,
        flatten_revisit_groups,
        load_poses_from_json,
        map_pose_to_video_frame,
        _extract_frame_poses,
    )
    from eval_revisit_nmr import (
        _sample_gap_matched_temporal_baselines,
        _sample_short_temporal_pairs,
    )
    from presampled_pairs import load_pairs_json, select_for_video

    torch.cuda.amp.autocast(enabled=False)

    parser = argparse.ArgumentParser(
        description="Object identity NMR evaluation with GroundingDINO + SAM2"
    )
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--pose_json", type=str, default=None)
    parser.add_argument("--per_video_pose", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./object_consistency_results")
    parser.add_argument("--groundingdino_config", type=str, required=True)
    parser.add_argument("--groundingdino_checkpoint", type=str, required=True)
    parser.add_argument(
        "--sam2_config",
        type=str,
        required=True,
        help="SAM2 model config, e.g. configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument(
        "--clip_checkpoint",
        type=str,
        default=None,
        help="CLIP ViT-B-32 TorchScript checkpoint for semantic scoring",
    )
    parser.add_argument("--text_prompt", type=str, default=DEFAULT_TEXT_PROMPT)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--nms_iou", type=float, default=0.60)
    parser.add_argument("--max_objects", type=int, default=10)
    parser.add_argument("--max_candidates_per_concept", type=int, default=5)
    parser.add_argument("--min_area_ratio", type=float, default=0.002)
    parser.add_argument("--match_threshold", type=float, default=0.50)
    parser.add_argument("--appearance_weight", type=float, default=0.5)
    parser.add_argument("--semantic_weight", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--num_video_shards", type=int, default=1)
    parser.add_argument("--video_shard_id", type=int, default=0)
    parser.add_argument("--max_eval_pairs", type=int, default=100)
    parser.add_argument("--short_multiplier", type=float, default=1.0)
    parser.add_argument("--min_gap_ratio", type=float, default=0.2)
    parser.add_argument("--angle_tolerance", type=float, default=None,
                        help="Max yaw difference in degrees (default: adaptive)")
    parser.add_argument("--translation_tolerance", type=float, default=None,
                        help="Max translation difference in meters (default: adaptive)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pairs_json",
        type=str,
        default=None,
        help=(
            "Pre-sampled shared pairs JSON (revisit/baseline/short in "
            "video-frame space). When set, these pairs are used instead of "
            "per-video internal sampling."
        ),
    )
    args = parser.parse_args()

    if not args.pose_json and not args.per_video_pose:
        parser.error("Either --pose_json or --per_video_pose must be specified")
    if args.num_video_shards < 1:
        parser.error("--num_video_shards must be >= 1")
    if args.video_shard_id < 0 or args.video_shard_id >= args.num_video_shards:
        parser.error("--video_shard_id must be in [0, num_video_shards)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resumed_results = {}
    if args.resume:
        for file_name in ("results.json", "results_partial.json"):
            result_path = output_dir / file_name
            if not result_path.exists():
                continue
            with open(result_path) as input_file:
                data = json.load(input_file)
            if isinstance(data, dict) and "per_video" in data:
                resumed_results.update(data["per_video"])
                print(f"  Resumed {len(resumed_results)} videos from {file_name}")

    print("=" * 60)
    print("Loading GroundingDINO, SAM2, DINOv2, and CLIP...")
    print("=" * 60)
    dino_extractor = DINOv2FeatureExtractor(device=device)
    clip_extractor = None
    if args.clip_checkpoint:
        clip_extractor = CLIPFeatureExtractor(args.clip_checkpoint, device=device)

    evaluator = GroundedSamObjectEvaluator(
        groundingdino_config=args.groundingdino_config,
        groundingdino_checkpoint=args.groundingdino_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        sam2_config=args.sam2_config,
        dino_extractor=dino_extractor,
        clip_extractor=clip_extractor,
        device=device,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_iou=args.nms_iou,
        max_objects=args.max_objects,
        max_candidates_per_concept=args.max_candidates_per_concept,
        min_area_ratio=args.min_area_ratio,
        match_threshold=args.match_threshold,
        appearance_weight=args.appearance_weight,
        semantic_weight=args.semantic_weight,
    )

    video_dir = Path(args.video_dir)
    all_videos = sorted(video_dir.glob("*.mp4"))
    if not all_videos:
        all_videos = sorted(video_dir.glob("*/gen.mp4"))
    if not all_videos:
        all_videos = sorted(video_dir.glob("*/output.mp4"))
    if args.max_videos:
        all_videos = all_videos[: args.max_videos]

    total_videos = len(all_videos)
    all_videos = all_videos[args.video_shard_id :: args.num_video_shards]
    if not all_videos:
        print(f"No videos found in {video_dir}")
        sys.exit(1)

    if args.num_video_shards > 1:
        print(
            f"  Video shard {args.video_shard_id}/{args.num_video_shards}: "
            f"{len(all_videos)}/{total_videos} videos"
        )

    per_video_pose_mode = args.per_video_pose is not None

    def get_pose_and_pairs(pose_json_path: str):
        revisit_groups, num_poses = extract_revisit_pairs(
            pose_json_path,
            angle_tolerance=args.angle_tolerance,
            translation_tolerance=args.translation_tolerance,
        )
        pose_data, sorted_keys, _ = load_poses_from_json(pose_json_path)
        frame_poses = _extract_frame_poses(pose_data, sorted_keys)
        return flatten_revisit_groups(revisit_groups), num_poses, frame_poses

    global_pairs = None
    global_num_poses = None
    global_frame_poses = None
    if not per_video_pose_mode:
        global_pairs, global_num_poses, global_frame_poses = get_pose_and_pairs(args.pose_json)
        if not global_pairs and args.pairs_json is None:
            print("No revisit pairs found.")
            sys.exit(1)

    presampled_pairs = None
    if args.pairs_json:
        presampled_pairs = load_pairs_json(args.pairs_json)
        print(
            f"  Using pre-sampled pairs from {args.pairs_json}: "
            f"revisit={len(presampled_pairs['revisit'])}, "
            f"baseline={len(presampled_pairs['baseline'])}, "
            f"short={len(presampled_pairs['short'])}"
        )

    per_video_results = dict(resumed_results)
    for video_index, video_path in enumerate(all_videos):
        evaluator.frame_cache.clear()

        video_name = (
            video_path.parent.name
            if video_path.name in ("gen.mp4", "output.mp4")
            else video_path.stem
        )
        if video_name in resumed_results:
            print(f"\n[{video_index + 1}/{len(all_videos)}] {video_name} - skipped (resumed)")
            continue

        if per_video_pose_mode:
            pose_path = video_path.parent / args.per_video_pose
            if not pose_path.exists():
                print(f"\n[{video_index + 1}/{len(all_videos)}] {video_name} - skipped (no pose file)")
                continue
            revisit_pairs, num_poses, frame_poses = get_pose_and_pairs(str(pose_path))
        else:
            revisit_pairs = global_pairs
            num_poses = global_num_poses
            frame_poses = global_frame_poses

        try:
            video_reader = VideoReader(str(video_path), ctx=decord_cpu(0))
        except Exception as exc:
            print(f"\n[{video_index + 1}/{len(all_videos)}] {video_name} - cannot open video: {exc}")
            continue

        num_video_frames = len(video_reader)
        effective_frames = min(num_video_frames, num_poses) if num_poses else num_video_frames

        if presampled_pairs is not None:
            revisit_video_pairs, baseline_pairs, short_pairs = select_for_video(
                presampled_pairs,
                num_video_frames,
                max_per_type=args.max_eval_pairs,
            )
            if not revisit_video_pairs:
                print(
                    f"\n[{video_index + 1}/{len(all_videos)}] "
                    f"{video_name} - no valid pre-sampled revisit pairs"
                )
                del video_reader
                continue

            baseline_stats = {"source": "presampled", "num_baseline": len(baseline_pairs)}
            short_stats = {"source": "presampled", "num_short": len(short_pairs)}
            print(
                f"\n[{video_index + 1}/{len(all_videos)}] {video_name} [pre-sampled]: "
                f"rev={len(revisit_video_pairs)}, base={len(baseline_pairs)}, short={len(short_pairs)}"
            )
        else:
            min_gap = int(effective_frames * args.min_gap_ratio)
            filtered_revisit = []
            for frame_a, frame_b, delta_position, delta_rotation in revisit_pairs:
                video_frame_a = map_pose_to_video_frame(frame_a, num_poses, num_video_frames)
                video_frame_b = map_pose_to_video_frame(frame_b, num_poses, num_video_frames)
                if video_frame_a >= effective_frames or video_frame_b >= effective_frames:
                    continue
                if abs(video_frame_b - video_frame_a) >= min_gap:
                    filtered_revisit.append((frame_a, frame_b, delta_position, delta_rotation))

            if not filtered_revisit:
                print(f"\n[{video_index + 1}/{len(all_videos)}] {video_name} - no valid revisit pairs")
                del video_reader
                continue

            if args.max_eval_pairs and len(filtered_revisit) > args.max_eval_pairs:
                filtered_revisit = random.sample(filtered_revisit, args.max_eval_pairs)

            revisit_video_pairs = [
                (
                    map_pose_to_video_frame(frame_a, num_poses, num_video_frames),
                    map_pose_to_video_frame(frame_b, num_poses, num_video_frames),
                )
                for frame_a, frame_b, _, _ in filtered_revisit
            ]

            exclusion_radius = max(10, effective_frames // 20)
            baseline_pairs, baseline_stats = _sample_gap_matched_temporal_baselines(
                revisit_video_pairs,
                frame_poses,
                effective_frames,
                num_temporal_bins=10,
                exclusion_radius=exclusion_radius,
                seed=args.seed,
            )
            short_count = min(100, max(20, int(len(revisit_video_pairs) * args.short_multiplier)))
            short_pairs, short_stats = _sample_short_temporal_pairs(
                frame_poses,
                effective_frames,
                count=short_count,
                min_gap=2,
                seed=args.seed,
            )

            print(
                f"\n[{video_index + 1}/{len(all_videos)}] {video_name}: "
                f"rev={len(revisit_video_pairs)}, base={len(baseline_pairs)}, short={len(short_pairs)}"
            )
            print(f"  baseline_stats={baseline_stats}")
            print(f"  short_stats={short_stats}")

        revisit_results = evaluate_object_pairs(
            video_reader,
            revisit_video_pairs,
            evaluator,
            label="revisit",
        )
        baseline_results = evaluate_object_pairs(
            video_reader,
            baseline_pairs,
            evaluator,
            label="baseline",
        )
        short_results = evaluate_object_pairs(
            video_reader,
            short_pairs,
            evaluator,
            label="short",
        )
        del video_reader

        revisit_aggregate = aggregate_object_metrics(revisit_results)
        baseline_aggregate = aggregate_object_metrics(baseline_results)
        short_aggregate = aggregate_object_metrics(short_results)
        nmr = compute_object_nmr(revisit_aggregate, baseline_aggregate, short_aggregate)

        per_video_results[video_name] = {
            "num_revisit_pairs": len(revisit_video_pairs),
            "num_baseline_pairs": len(baseline_pairs),
            "num_short_pairs": len(short_pairs),
            "baseline_stats": baseline_stats,
            "short_stats": short_stats,
            "revisit": revisit_aggregate,
            "baseline": baseline_aggregate,
            "short": short_aggregate,
            "nmr": nmr,
            "pair_results": {
                "revisit": revisit_results,
                "baseline": baseline_results,
                "short": short_results,
            },
        }

        print(f"  {'Metric':<28} {'Rev':>8} {'Base':>8} {'Short':>8} {'NMR':>8}")
        for key in [
            "object_existence_rate",
            "object_appearance_similarity",
            "object_semantic_similarity",
            "object_appearance_persistence",
            "object_semantic_persistence",
        ]:
            revisit_value = revisit_aggregate.get(key, {}).get("mean", 0.0)
            baseline_value = baseline_aggregate.get(key, {}).get("mean", 0.0)
            short_value = short_aggregate.get(key, {}).get("mean", 0.0)
            nmr_value = nmr.get(key, {}).get("value", None)
            nmr_text = "--" if nmr_value is None else f"{nmr_value:.4f}"
            print(
                f"  {key:<28} {revisit_value:>8.4f} "
                f"{baseline_value:>8.4f} {short_value:>8.4f} {nmr_text:>8}"
            )

        save_object_results(output_dir, per_video_results)

    final_payload = {
        "global_summary": build_object_global_summary(per_video_results),
        "per_video": per_video_results,
    }
    with open(output_dir / "results.json", "w") as output_file:
        json.dump(final_payload, output_file, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output_dir / 'results.json'}")

if __name__ == "__main__":
    run_object_identity_cli()
