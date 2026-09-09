"""
Revisit Consistency Evaluation - Unified Script
=================================================
输入视频目录和 pose JSON，自动检测 revisit 帧对，
计算 DINO、BoQ、PSNR、SSIM 指标，并输出相关性分析。

Usage:
    python eval_revisit.py \
        --video_dir /path/to/videos \
        --pose_json /path/to/trajectory.json \
        --output_dir ./eval_results \
        --angle_tolerance 2.0 \
        --translation_tolerance 0.1 \
        --device cuda:0
"""

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader, cpu as decord_cpu
from PIL import Image
from scipy.stats import spearmanr
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from torchvision.transforms import InterpolationMode

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _disable_tqdm_progress_bars() -> None:
    """Disable tqdm progress bars so redirected evaluation logs stay readable."""
    try:
        import tqdm as tqdm_module
    except ImportError:
        return

    original_tqdm = tqdm_module.tqdm

    def quiet_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return original_tqdm(*args, **kwargs)

    tqdm_module.tqdm = quiet_tqdm

_disable_tqdm_progress_bars()

# ==============================================================================
# Pose loading and revisit detection
# (Aligned with eval_revisit_metrics.py: yaw + tz, segment-based matching)
# ==============================================================================

import math


def _circular_yaw_distance_deg(a: float, b: float) -> float:
    """Return the shortest angular distance between two yaw angles."""
    return abs((a - b + 180.0) % 360.0 - 180.0)

def load_poses_from_json(pose_json_path: str):
    """Load pose data from JSON. Returns (pose_data_dict, total_frames)."""
    if not os.path.isfile(pose_json_path):
        raise FileNotFoundError(
            f"Pose/trajectory JSON not found: {pose_json_path}. "
            f"Check the TASK pose_json path in your run config (e.g. run_config.local.conf)."
        )
    with open(pose_json_path, "r") as f:
        pose_data = json.load(f)
    sorted_keys = sorted(pose_data.keys(), key=lambda x: int(x))
    total_frames = len(sorted_keys)
    return pose_data, sorted_keys, total_frames

def _extract_frame_poses(pose_data, sorted_keys):
    """Extract per-frame yaw angle, full 3D translation from extrinsic matrices."""
    frame_poses = {}
    for key in sorted_keys:
        idx = int(key)
        ext = pose_data[key]["extrinsic"]
        r02, r00 = ext[0][2], ext[0][0]
        yaw_deg = math.atan2(r02, r00) * 180.0 / math.pi
        tx, ty, tz = ext[0][3], ext[1][3], ext[2][3]
        frame_poses[idx] = {"yaw": yaw_deg, "tx": tx, "ty": ty, "tz": tz}
    return frame_poses

def _detect_motion_segments(frame_poses, sorted_keys):
    """Detect motion segments from frame poses using full 3D translation + yaw."""
    keys = [int(k) for k in sorted_keys]
    motion_phases = []
    current_phase = None
    phase_start = 0

    for i in range(1, len(keys)):
        idx = keys[i]
        prev_idx = keys[i - 1]
        cur = frame_poses[idx]
        prev = frame_poses[prev_idx]

        dtx = cur["tx"] - prev["tx"]
        dty = cur["ty"] - prev["ty"]
        dtz = cur["tz"] - prev["tz"]
        dyaw = cur["yaw"] - prev["yaw"]
        translation_dist = math.sqrt(dtx * dtx + dty * dty + dtz * dtz)

        if translation_dist > 0.01 and abs(dyaw) < 0.5:
            # Determine primary translation direction
            abs_vals = {"tx": abs(dtx), "ty": abs(dty), "tz": abs(dtz)}
            primary_axis = max(abs_vals, key=abs_vals.get)
            primary_val = {"tx": dtx, "ty": dty, "tz": dtz}[primary_axis]
            motion = "W" if primary_val > 0 else "S"
        elif abs(dyaw) > 0.5:
            motion = "R" if dyaw > 0 else "L"
        else:
            motion = "."

        if motion != current_phase and motion != ".":
            if current_phase is not None:
                motion_phases.append((phase_start, keys[i - 1], current_phase))
            current_phase = motion
            phase_start = idx

    if current_phase is not None:
        motion_phases.append((phase_start, keys[-1], current_phase))

    return motion_phases

def _find_revisit_pairs(frame_poses, segments, angle_tol, trans_tol, min_frame_gap=10):
    """Find revisit pairs between all segment combinations using full 3D position.

    Pairs with frame gap < min_frame_gap are filtered out as they are unlikely
    to represent true revisits.
    """
    revisit_groups = []

    for later_idx in range(1, len(segments)):
        later_start, later_end, later_type = segments[later_idx]
        for earlier_idx in range(later_idx):
            earlier_start, earlier_end, earlier_type = segments[earlier_idx]

            pairs = []
            for later_frame in range(later_start, later_end + 1):
                lp = frame_poses[later_frame]
                best_match = None
                best_diff = float("inf")

                for earlier_frame in range(earlier_start, earlier_end + 1):
                    if abs(later_frame - earlier_frame) < min_frame_gap:
                        continue

                    ep = frame_poses[earlier_frame]

                    yaw_diff = _circular_yaw_distance_deg(lp["yaw"], ep["yaw"])
                    pos_diff = math.sqrt(
                        (lp["tx"] - ep["tx"]) ** 2 +
                        (lp["ty"] - ep["ty"]) ** 2 +
                        (lp["tz"] - ep["tz"]) ** 2
                    )
                    total_diff = yaw_diff + pos_diff * 10  # weighted

                    if yaw_diff <= angle_tol and pos_diff <= trans_tol:
                        if total_diff < best_diff:
                            best_diff = total_diff
                            best_match = earlier_frame

                if best_match is not None:
                    pairs.append((later_frame, best_match))

            if pairs:
                group_name = f"Seg{later_idx}({later_type})↔Seg{earlier_idx}({earlier_type})"
                revisit_groups.append({"name": group_name, "pairs": pairs})

    return revisit_groups

def _extract_full_poses(pose_data, sorted_keys):
    """Extract full 3D position and rotation from extrinsic matrices."""
    full_poses = {}
    for key in sorted_keys:
        idx = int(key)
        ext = pose_data[key]["extrinsic"]
        tx, ty, tz = ext[0][3], ext[1][3], ext[2][3]
        # Extract rotation as 3x3 matrix for comparison
        rot = [ext[r][:3] for r in range(3)]
        full_poses[idx] = {"tx": tx, "ty": ty, "tz": tz, "rot": rot}
    return full_poses


def _rotation_distance_deg(rot_a, rot_b):
    """Compute rotation distance in degrees between two 3x3 rotation matrices."""
    trace = sum(rot_a[i][j] * rot_b[i][j] for i in range(3) for j in range(3))
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.acos(cos_angle) * 180.0 / math.pi


def _find_revisit_pairs_3d(
    poses,
    sorted_keys,
    position_threshold,
    rotation_threshold_deg,
    min_frame_gap=10,
):
    """Find revisit pairs using 3D position and circular yaw distance.
    Fallback method when segment-based detection fails."""
    keys = [int(k) for k in sorted_keys]
    pairs = []
    seen = set()

    for i in range(len(keys)):
        for j in range(i + min_frame_gap, len(keys)):
            fi, fj = keys[i], keys[j]
            pi, pj = poses[fi], poses[fj]

            dx = pi["tx"] - pj["tx"]
            dy = pi["ty"] - pj["ty"]
            dz = pi["tz"] - pj["tz"]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist > position_threshold:
                continue

            yaw_dist = _circular_yaw_distance_deg(pi["yaw"], pj["yaw"])
            if yaw_dist > rotation_threshold_deg:
                continue

            key = (fi, fj)
            if key not in seen:
                # Group format is consistently (later_frame, earlier_frame).
                pairs.append((fj, fi))
                seen.add(key)

    # Convert to group format
    if pairs:
        return [{"name": "3D_position_match", "pairs": pairs}]
    return []


def _trim_static_tail(pose_data, sorted_keys, move_threshold=0.01, rot_threshold_deg=0.5):
    """Remove trailing static frames where the camera is not moving or rotating.

    Scans from the end backwards, removing frames whose 3D position AND rotation
    are identical (within threshold) to the previous frame.

    Returns:
        trimmed sorted_keys (list of str), number of frames removed
    """
    if len(sorted_keys) < 3:
        return sorted_keys, 0

    keys = list(sorted_keys)
    last_moving_idx = len(keys) - 1

    for i in range(len(keys) - 1, 0, -1):
        ext_cur = pose_data[keys[i]]["extrinsic"]
        ext_prev = pose_data[keys[i - 1]]["extrinsic"]

        # Check translation change
        dx = ext_cur[0][3] - ext_prev[0][3]
        dy = ext_cur[1][3] - ext_prev[1][3]
        dz = ext_cur[2][3] - ext_prev[2][3]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Check rotation change
        rot_a = [ext_cur[r][:3] for r in range(3)]
        rot_b = [ext_prev[r][:3] for r in range(3)]
        rot_dist = _rotation_distance_deg(rot_a, rot_b)

        if dist > move_threshold or rot_dist > rot_threshold_deg:
            last_moving_idx = i
            break

    trimmed_count = len(keys) - 1 - last_moving_idx
    if trimmed_count > 0:
        keys = keys[:last_moving_idx + 1]

    return keys, trimmed_count


def _compute_adaptive_thresholds(frame_poses, sorted_keys):
    """Compute adaptive revisit thresholds based on per-frame motion statistics.

    Uses the P95 of consecutive-frame translation and rotation deltas as the
    baseline "one step" motion magnitude. Revisit tolerance is set to twice
    this baseline, subject to fixed safety bounds, so that quantization and
    small integration errors do not suppress genuine returns.

    Returns:
        (adaptive_angle_tol, adaptive_trans_tol)
    """
    keys = [int(k) for k in sorted_keys]
    trans_deltas = []
    yaw_deltas = []

    for i in range(1, len(keys)):
        cur = frame_poses[keys[i]]
        prev = frame_poses[keys[i - 1]]
        dt = math.sqrt(
            (cur["tx"] - prev["tx"]) ** 2
            + (cur["ty"] - prev["ty"]) ** 2
            + (cur["tz"] - prev["tz"]) ** 2
        )
        dy = _circular_yaw_distance_deg(cur["yaw"], prev["yaw"])
        trans_deltas.append(dt)
        yaw_deltas.append(dy)

    if not trans_deltas:
        return 2.0, 0.1

    p95_trans = float(np.percentile(trans_deltas, 95))
    p95_yaw = float(np.percentile(yaw_deltas, 95))

    # Revisit tolerance = 2× one-step motion magnitude (P95), with sensible bounds
    adaptive_trans_tol = max(0.1, min(2.0 * p95_trans, 1.0))
    adaptive_angle_tol = max(1.0, min(2.0 * p95_yaw, 5.0))

    return adaptive_angle_tol, adaptive_trans_tol


def extract_revisit_pairs(
    pose_json_path,
    angle_tolerance=None,
    translation_tolerance=None,
    return_thresholds=False,
):
    """Extract revisit frame pairs from trajectory JSON.

    A revisit pair is two frames that share (nearly) the same camera pose
    but appear at different points in the trajectory.

    When angle_tolerance / translation_tolerance are None (default), adaptive
    thresholds are computed from the per-frame motion statistics of the
    trajectory.  Pass explicit values to override.

    Uses segment-based detection first; falls back to full 3D position matching
    if no pairs are found. Trailing static frames are trimmed before detection.

    Returns:
        revisit_groups: list of dicts with 'name' and 'pairs' (later_frame, earlier_frame)
        total_trajectory_frames: number of frames in trajectory (before trimming)
        When return_thresholds is true, a third dictionary records the resolved
        threshold mode and values used for pair detection.
    """
    pose_data, sorted_keys, total_frames = load_poses_from_json(pose_json_path)
    angle_is_adaptive = angle_tolerance is None
    translation_is_adaptive = translation_tolerance is None

    # Trim trailing static frames to avoid spurious matches
    trimmed_keys, trimmed_count = _trim_static_tail(pose_data, sorted_keys)
    if trimmed_count > 0:
        print(f"  Trimmed {trimmed_count} static trailing frames ({len(sorted_keys)} -> {len(trimmed_keys)})")

    frame_poses = _extract_frame_poses(pose_data, trimmed_keys)

    # Adaptive thresholds from per-frame motion statistics
    if angle_tolerance is None or translation_tolerance is None:
        auto_angle, auto_trans = _compute_adaptive_thresholds(frame_poses, trimmed_keys)
        if angle_tolerance is None:
            angle_tolerance = auto_angle
        if translation_tolerance is None:
            translation_tolerance = auto_trans
        print(f"  Adaptive thresholds: angle_tol={angle_tolerance:.2f}°, trans_tol={translation_tolerance:.4f}m")

    segments = _detect_motion_segments(frame_poses, trimmed_keys)
    seg_groups = _find_revisit_pairs(
        frame_poses, segments, angle_tolerance, translation_tolerance
    )
    seg_count = sum(len(g["pairs"]) for g in seg_groups) if seg_groups else 0

    # Also run 3D matching if segment-based found few pairs (< 3)
    min_seg_pairs = 3
    if seg_count < min_seg_pairs:
        if seg_count > 0:
            print(f"  Segment-based found only {seg_count} pairs (< {min_seg_pairs}), also trying 3D matching...")
        else:
            print("  Segment-based detection found no pairs, trying 3D position matching...")
        pos3d_groups = _find_revisit_pairs_3d(
            frame_poses,
            trimmed_keys,
            position_threshold=translation_tolerance,
            rotation_threshold_deg=angle_tolerance,
        )
        pos3d_count = sum(len(g["pairs"]) for g in pos3d_groups) if pos3d_groups else 0

        if pos3d_count > seg_count:
            print(f"  Using 3D position matching: {pos3d_count} pairs")
            revisit_groups = pos3d_groups
        else:
            revisit_groups = seg_groups
    else:
        revisit_groups = seg_groups

    if return_thresholds:
        if angle_is_adaptive and translation_is_adaptive:
            threshold_mode = "adaptive"
        elif angle_is_adaptive or translation_is_adaptive:
            threshold_mode = "mixed"
        else:
            threshold_mode = "fixed"
        threshold_info = {
            "mode": threshold_mode,
            "angle_tolerance_deg": float(angle_tolerance),
            "translation_tolerance_m": float(translation_tolerance),
        }
        return revisit_groups, len(trimmed_keys), threshold_info

    return revisit_groups, len(trimmed_keys)


def flatten_revisit_groups(revisit_groups):
    """Convert revisit groups to flat list of (earlier_frame, later_frame, 0.0, 0.0) tuples.

    This maintains compatibility with the rest of the evaluation pipeline.
    """
    flat_pairs = []
    seen = set()
    for group in revisit_groups:
        for later_frame, earlier_frame in group["pairs"]:
            key = (earlier_frame, later_frame)
            if key not in seen:
                flat_pairs.append((earlier_frame, later_frame, 0.0, 0.0))
                seen.add(key)
    flat_pairs.sort(key=lambda x: (x[1], x[0]))
    return flat_pairs


def map_pose_to_video_frame(pose_idx: int, n_pose: int, n_video: int) -> int:
    """Map pose frame index to video frame index.

    When n_pose == n_video, identity mapping.
    Otherwise, use direct 1:1 mapping (pose_idx == video_idx).
    Pairs with pose_idx >= n_video will be skipped in evaluate_video.
    """
    return pose_idx

# ==============================================================================
# Metric extractors (moved into the metrics/ package, imported here for the
# evaluate_video / main path below and for backward-compatible re-use).
# ==============================================================================

from metrics import (
    DINOv2FeatureExtractor,
    BoQFeatureExtractor,
    MutualVPRFeatureExtractor,
    KeypointMatcher,
    LpipsClipComputer,
    compute_psnr_ssim,
)


def _get_effective_video_frame_count(video_path: str, max_pose_frames: Optional[int] = None) -> int:
    """Return video frame count capped by trim-derived pose frame count."""
    vr = VideoReader(video_path, ctx=decord_cpu(0))
    video_frame_count = len(vr)
    if max_pose_frames is None:
        return video_frame_count
    return min(video_frame_count, max_pose_frames)


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_video(
    video_path: str,
    revisit_pairs: List[Tuple[int, int, float, float]],
    n_pose: int,
    dino_ext=None,
    boq_ext=None,
    mvpr_ext=None,
    keypoint_matcher: Optional[KeypointMatcher] = None,
    skip_pixel_metrics: bool = False,
    lpips_clip: Optional[LpipsClipComputer] = None,
    max_video_frames: Optional[int] = None,
    max_eval_pairs: int = 0,
) -> dict:
    """Evaluate all revisit pairs for a single video.

    Returns:
        dict with keys:
          - "pair_results": list of per-pair result dicts
          - "clip_video": float or None (CLIP-Video score, only if lpips_clip enabled)
    """
    vr = VideoReader(video_path, ctx=decord_cpu(0))
    n_video = len(vr)
    effective_video_frames = n_video
    if max_video_frames is not None:
        effective_video_frames = min(n_video, max_video_frames)

    sampled_pairs = revisit_pairs
    if max_eval_pairs and len(revisit_pairs) > max_eval_pairs:
        sampled_pairs = random.sample(revisit_pairs, max_eval_pairs)
        print(f"      Sampled {max_eval_pairs}/{len(revisit_pairs)} pairs for evaluation", flush=True)

    results = []

    for pair_idx, (fa, fb, dpos, drot) in enumerate(sampled_pairs):
        vid_a = map_pose_to_video_frame(fa, n_pose, n_video)
        vid_b = map_pose_to_video_frame(fb, n_pose, n_video)

        if vid_a >= effective_video_frames or vid_b >= effective_video_frames:
            results.append({"status": "skipped", "pose_frames": [fa, fb], "video_frames": [vid_a, vid_b]})
            continue

        img_a = vr[vid_a].asnumpy()
        img_b = vr[vid_b].asnumpy()

        # Pixel-level metrics (optional, slow)
        if not skip_pixel_metrics:
            _pix = compute_psnr_ssim(img_a, img_b)

        # Feature-level metrics
        result_entry = {
            "status": "ok",
            "pose_frames": [fa, fb],
            "video_frames": [vid_a, vid_b],
            "dpos": round(dpos, 6),
            "drot": round(drot, 4),
        }
        if dino_ext is not None:
            dino_sim = dino_ext.similarity(img_a, img_b)
            result_entry["dino_similarity"] = round(dino_sim, 6)
        if boq_ext is not None:
            boq_sim = boq_ext.similarity(img_a, img_b)
            result_entry["boq_similarity"] = round(boq_sim, 6)
        if mvpr_ext is not None:
            mvpr_sim = mvpr_ext.similarity(img_a, img_b)
            result_entry["mvpr_similarity"] = round(mvpr_sim, 6)
        if not skip_pixel_metrics:
            result_entry["psnr"] = _pix["psnr"]
            result_entry["ssim"] = _pix["ssim"]

        # LPIPS (optional)
        if lpips_clip is not None:
            lpips_val = lpips_clip.compute_lpips(img_a, img_b)
            result_entry["lpips"] = round(lpips_val, 6)

        # Keypoint matching metrics
        if keypoint_matcher is not None:
            kp_result = keypoint_matcher.match(img_a, img_b)
            result_entry["num_keypoints_a"] = kp_result["num_keypoints_a"]
            result_entry["num_keypoints_b"] = kp_result["num_keypoints_b"]
            result_entry["num_matches"] = kp_result["num_matches"]
            result_entry["match_ratio"] = kp_result["match_ratio"]

        results.append(result_entry)

        if (pair_idx + 1) % 10 == 0:
            print(f"      {pair_idx+1}/{len(sampled_pairs)} pairs done", flush=True)

    # CLIP-Video score (computed over effective frames, not just revisit pairs)
    clip_video_score = None
    if lpips_clip is not None:
        all_frames_rgb = [vr[i].asnumpy() for i in range(effective_video_frames)]
        clip_video_score = lpips_clip.compute_clip_video_score(all_frames_rgb)
        del all_frames_rgb

    return {"pair_results": results, "clip_video": clip_video_score}


def compute_correlations(all_results: dict) -> dict:
    """Compute Pearson and Spearman correlations between all metrics."""
    psnr_vals, ssim_vals, dino_vals, boq_vals = [], [], [], []

    for video_name, video_results in all_results.items():
        if not isinstance(video_results, list):
            continue
        for r in video_results:
            if r.get("status") != "ok":
                continue
            psnr_vals.append(r["psnr"])
            ssim_vals.append(r["ssim"])
            dino_vals.append(r["dino_similarity"])
            boq_vals.append(r["boq_similarity"])

    if len(psnr_vals) < 3:
        return {}

    psnr_arr = np.array(psnr_vals)
    ssim_arr = np.array(ssim_vals)
    dino_arr = np.array(dino_vals)
    boq_arr = np.array(boq_vals)

    correlations = {
        "num_measurements": len(psnr_vals),
        "pearson": {
            "dino_vs_psnr": round(float(np.corrcoef(dino_arr, psnr_arr)[0, 1]), 4),
            "boq_vs_psnr": round(float(np.corrcoef(boq_arr, psnr_arr)[0, 1]), 4),
            "dino_vs_ssim": round(float(np.corrcoef(dino_arr, ssim_arr)[0, 1]), 4),
            "boq_vs_ssim": round(float(np.corrcoef(boq_arr, ssim_arr)[0, 1]), 4),
            "dino_vs_boq": round(float(np.corrcoef(dino_arr, boq_arr)[0, 1]), 4),
        },
        "spearman": {
            "dino_vs_psnr": round(float(spearmanr(dino_arr, psnr_arr).correlation), 4),
            "boq_vs_psnr": round(float(spearmanr(boq_arr, psnr_arr).correlation), 4),
            "dino_vs_ssim": round(float(spearmanr(dino_arr, ssim_arr).correlation), 4),
            "boq_vs_ssim": round(float(spearmanr(boq_arr, ssim_arr).correlation), 4),
            "dino_vs_boq": round(float(spearmanr(dino_arr, boq_arr).correlation), 4),
        },
    }
    return correlations


def visualize_pairs(
    video_path: str,
    revisit_pairs: List[Tuple[int, int, float, float]],
    video_results: List[dict],
    n_pose: int,
    output_dir: Path,
    max_pairs: int = 10,
):
    """Visualize sample revisit pairs for a video."""
    vr = VideoReader(str(video_path), ctx=decord_cpu(0))
    n_video = len(vr)
    if Path(video_path).name in ("gen.mp4", "output.mp4"):
        video_name = Path(video_path).parent.name
    else:
        video_name = Path(video_path).stem

    vis_dir = output_dir / "visualizations" / video_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Select pairs to visualize (sample diverse ones)
    valid_indices = [i for i, r in enumerate(video_results) if r.get("status") == "ok"]
    sample_count = min(max_pairs, len(valid_indices))
    if sample_count == 0:
        return

    sampled = random.sample(valid_indices, sample_count)

    for idx in sampled:
        r = video_results[idx]
        fa, fb = r["pose_frames"]
        vid_a, vid_b = r["video_frames"]

        img_a = vr[vid_a].asnumpy()
        img_b = vr[vid_b].asnumpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        metric_parts = [f"Pose {fa} <-> {fb} (dpos={r['dpos']:.4f}, drot={r['drot']:.2f}°)"]
        detail_parts = []
        if 'psnr' in r:
            detail_parts.append(f"PSNR={r['psnr']:.2f}")
        if 'ssim' in r:
            detail_parts.append(f"SSIM={r['ssim']:.4f}")
        if 'dino_similarity' in r:
            detail_parts.append(f"DINO={r['dino_similarity']:.4f}")
        if 'boq_similarity' in r:
            detail_parts.append(f"BoQ={r['boq_similarity']:.4f}")
        if 'lpips' in r:
            detail_parts.append(f"LPIPS={r['lpips']:.4f}")
        if 'num_matches' in r:
            detail_parts.append(f"KP={r['num_matches']}({r['match_ratio']:.2f})")
        if detail_parts:
            metric_parts.append(" | ".join(detail_parts))
        title = "\n".join(metric_parts)
        fig.suptitle(title, fontsize=10, fontweight="bold")

        axes[0].imshow(img_a)
        axes[0].set_title(f"Pose Frame {fa} (Video Frame {vid_a})", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(img_b)
        axes[1].set_title(f"Pose Frame {fb} (Video Frame {vid_b})", fontsize=9)
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(vis_dir / f"pair_{idx:03d}_f{fa}_vs_f{fb}.png", dpi=120, bbox_inches="tight")
        plt.close()


# ==============================================================================
# Main
# ==============================================================================

def _collect_global_summary(
    all_results,
    all_clip_video_scores,
    total_revisit_pairs_count,
):
    """Collect global summary metrics from all video results.

    Args:
        all_results: dict of video_name -> list of per-pair result dicts
        all_clip_video_scores: dict of video_name -> float (CLIP-Video score)
        total_revisit_pairs_count: int
    """
    all_psnr, all_ssim, all_dino, all_boq, all_mvpr, all_lpips = [], [], [], [], [], []
    all_num_matches, all_match_ratio = [], []
    total_measurements = 0

    for video_results in all_results.values():
        if not isinstance(video_results, list):
            continue
        for r in video_results:
            if r.get("status") != "ok":
                continue
            total_measurements += 1
            if "psnr" in r:
                all_psnr.append(r["psnr"])
            if "ssim" in r:
                all_ssim.append(r["ssim"])
            if "dino_similarity" in r:
                all_dino.append(r["dino_similarity"])
            if "boq_similarity" in r:
                all_boq.append(r["boq_similarity"])
            if "mvpr_similarity" in r:
                all_mvpr.append(r["mvpr_similarity"])
            if "lpips" in r:
                all_lpips.append(r["lpips"])
            if "num_matches" in r:
                all_num_matches.append(r["num_matches"])
            if "match_ratio" in r:
                all_match_ratio.append(r["match_ratio"])

    global_summary = {
        "num_measurements": total_measurements,
        "num_videos": len(all_results),
        "num_revisit_pairs": total_revisit_pairs_count,
    }

    def _stat(vals):
        return {"mean": round(np.mean(vals), 6), "std": round(np.std(vals), 6),
                "median": round(np.median(vals), 6)}

    if all_dino:
        global_summary["dino_similarity"] = _stat(all_dino)
    if all_boq:
        global_summary["boq_similarity"] = _stat(all_boq)
    if all_mvpr:
        global_summary["mvpr_similarity"] = _stat(all_mvpr)
    if all_psnr:
        global_summary["psnr"] = _stat(all_psnr)
    if all_ssim:
        global_summary["ssim"] = _stat(all_ssim)
    if all_lpips:
        global_summary["lpips"] = _stat(all_lpips)
    if all_num_matches:
        global_summary["num_matches"] = {"mean": round(np.mean(all_num_matches), 2),
                                         "std": round(np.std(all_num_matches), 2),
                                         "median": round(np.median(all_num_matches), 2)}
    if all_match_ratio:
        global_summary["match_ratio"] = _stat(all_match_ratio)

    # CLIP-Video metric is per-video, not per-pair
    clip_video_vals = [v for v in all_clip_video_scores.values() if v is not None]
    if clip_video_vals:
        global_summary["clip_video"] = _stat(clip_video_vals)

    return global_summary, all_psnr


def _print_summary(global_summary, correlations, total_revisit_pairs_count):
    """Print summary table."""
    print(f"\n  Total measurements: {global_summary.get('num_measurements', 0)}")
    print(f"  Videos: {global_summary.get('num_videos', 0)} | "
          f"Revisit pairs: {total_revisit_pairs_count}")
    print(f"\n  {'Metric':<20} {'Mean':>10} {'Std':>10} {'Median':>10}")
    print(f"  {'-'*50}")
    for metric in [
        "dino_similarity", "boq_similarity", "mvpr_similarity",
        "psnr", "ssim", "lpips", "clip_video",
        "num_matches", "match_ratio",
    ]:
        if metric not in global_summary:
            continue
        m = global_summary[metric]
        print(f"  {metric:<20} {m['mean']:>10.4f} {m['std']:>10.4f} {m['median']:>10.4f}")

    if correlations:
        print(f"\n  {'='*50}")
        print(f"  Pearson Correlation")
        print(f"  {'='*50}")
        for key, val in correlations.get("pearson", {}).items():
            print(f"    {key:<20}: r = {val:.4f}")
        print(f"\n  {'='*50}")
        print(f"  Spearman Rank Correlation")
        print(f"  {'='*50}")
        for key, val in correlations.get("spearman", {}).items():
            print(f"    {key:<20}: rho = {val:.4f}")


def _write_json_atomic(output_path: Path, payload: dict) -> None:
    """Write JSON atomically to avoid corrupted resume files after interruption."""
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temp_path, "w") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
    os.replace(temp_path, output_path)

def save_partial_results(output_dir: Path, per_video_results: dict,
                         clip_video_scores: dict):
    """Save resumable partial results using the structured results.json layout."""
    partial_output = {
        "per_video_results": per_video_results,
        "clip_video_scores": clip_video_scores if clip_video_scores else None,
    }
    _write_json_atomic(output_dir / "results_partial.json", partial_output)


def main():
    parser = argparse.ArgumentParser(description="Revisit Consistency Evaluation (DINO + BoQ + LightGlue)")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files (.mp4 or */gen.mp4)")
    parser.add_argument("--pose_json", type=str, default=None,
                        help="Global trajectory pose JSON (for shared-pose mode)")
    parser.add_argument("--per_video_pose", type=str, default=None,
                        help="Per-video pose filename, e.g. 'pose_prepared.json'. "
                             "Each video subdir must contain this file.")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Output directory for results")
    parser.add_argument("--angle_tolerance", type=float, default=None,
                        help="Max yaw difference in degrees (default: adaptive)")
    parser.add_argument("--translation_tolerance", type=float, default=None,
                        help="Max translation difference in meters (default: adaptive)")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Max number of videos to evaluate")
    parser.add_argument("--max_vis_pairs", type=int, default=10,
                        help="Max pairs to visualize per video")
    parser.add_argument("--no_vis", action="store_true", help="Disable visualization")
    parser.add_argument("--skip_pixel_metrics", action="store_true",
                        help="Skip PSNR/SSIM computation (much faster)")
    parser.add_argument("--only_keypoints", action="store_true",
                        help="Only run keypoint matching (skip DINO/BoQ/PSNR/SSIM)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous partial results (skip already evaluated videos)")
    parser.add_argument("--enable_lpips_clip", action="store_true",
                        help="Enable LPIPS (per-pair) and CLIP-Video (per-video) metrics")
    parser.add_argument("--clip_checkpoint", type=str,
                        default=os.path.expanduser("~/.cache/clip/ViT-B-32.pt"),
                        help="Path to CLIP ViT-B-32 TorchScript checkpoint")
    parser.add_argument("--max_eval_pairs", type=int, default=200,
                        help="Max revisit pairs to evaluate per video; randomly sample if exceeded (default: 200, 0=unlimited)")
    args = parser.parse_args()

    if not args.pose_json and not args.per_video_pose:
        parser.error("Either --pose_json or --per_video_pose must be specified")

    if args.only_keypoints:
        args.skip_pixel_metrics = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Resume: load previously completed results
    # =========================================================================
    resumed_results = {}
    resumed_clip_video_scores = {}

    def _merge_resume_data(source_path: Path) -> bool:
        nonlocal resumed_results, resumed_clip_video_scores
        if not source_path.exists():
            return False

        try:
            with open(source_path) as input_file:
                resume_data = json.load(input_file)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  Resume warning: ignoring unreadable {source_path.name}: {exc}")
            return False

        if isinstance(resume_data, dict) and "per_video_results" in resume_data:
            old_results = resume_data.get("per_video_results")
            if isinstance(old_results, dict):
                resumed_results.update(old_results)
            old_clip_video_scores = resume_data.get("clip_video_scores")
            if isinstance(old_clip_video_scores, dict):
                resumed_clip_video_scores.update(old_clip_video_scores)
        elif isinstance(resume_data, dict):
            resumed_results.update(resume_data)
        return True

    if args.resume:
        partial_path = output_dir / "results_partial.json"
        final_path = output_dir / "results.json"
        loaded_any_resume = False
        loaded_any_resume = _merge_resume_data(final_path) or loaded_any_resume
        loaded_any_resume = _merge_resume_data(partial_path) or loaded_any_resume

        if loaded_any_resume:
            print(
                f"  Resuming: loaded {len(resumed_results)} completed videos "
                f"({len(resumed_clip_video_scores)} clip_video scores)"
            )
        else:
            print("  Resume requested but no previous readable results found, starting fresh")

    # =========================================================================
    # Step 1: Load models
    # =========================================================================
    print("=" * 60)
    print("Step 1: Loading models...")
    print("=" * 60)

    device = args.device if torch.cuda.is_available() else "cpu"
    if args.only_keypoints:
        dino_ext = None
        boq_ext = None
        mvpr_ext = None
    else:
        dino_ext = DINOv2FeatureExtractor(device=device)
        boq_ext = BoQFeatureExtractor(device=device)
        mvpr_ext = MutualVPRFeatureExtractor(device=device)
    keypoint_matcher = KeypointMatcher(device=device)

    lpips_clip = None
    if args.enable_lpips_clip:
        lpips_clip = LpipsClipComputer(clip_checkpoint_path=args.clip_checkpoint, device=device)

    # =========================================================================
    # Step 2: Discover videos
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Discovering videos...")
    print("=" * 60)

    video_dir = Path(args.video_dir)
    all_videos = sorted(video_dir.glob("*.mp4"))
    if not all_videos:
        all_videos = sorted(video_dir.glob("*/gen.mp4"))
    if not all_videos:
        all_videos = sorted(video_dir.glob("*/output.mp4"))
    if not all_videos:
        print(f"  No .mp4 files found in {video_dir}")
        sys.exit(1)

    if args.max_videos:
        all_videos = all_videos[:args.max_videos]

    print(f"  Found {len(all_videos)} videos")

    # =========================================================================
    # Step 3: Evaluate
    # =========================================================================
    per_video_pose_mode = args.per_video_pose is not None
    all_results = dict(resumed_results)  # pre-fill with resumed data
    all_clip_video_scores = dict(resumed_clip_video_scores)  # pre-fill with resumed clip scores
    all_revisit_info = {}  # store per-video revisit pair info
    total_revisit_count = 0
    skipped_count = 0

    if not per_video_pose_mode:
        # ------ Global pose mode: detect pairs once, apply to all videos ------
        print("\n" + "=" * 60)
        print("Step 3a: Detecting revisit pairs (global pose)...")
        print("=" * 60)

        revisit_groups, n_pose = extract_revisit_pairs(
            args.pose_json,
            angle_tolerance=args.angle_tolerance,
            translation_tolerance=args.translation_tolerance,
        )
        print(f"  Loaded {n_pose} poses")
        total_pairs = sum(len(g["pairs"]) for g in revisit_groups)
        print(f"  Revisit groups: {len(revisit_groups)}, total pairs: {total_pairs}")
        for group in revisit_groups:
            print(f"    {group['name']}: {len(group['pairs'])} pairs")

        revisit_pairs = flatten_revisit_groups(revisit_groups)
        if not revisit_pairs:
            print("  ERROR: No revisit pairs detected!")
            sys.exit(1)
        total_revisit_count = len(revisit_pairs)

        print(f"\n  Flattened to {len(revisit_pairs)} unique pairs")

        print("\n" + "=" * 60)
        print("Step 3b: Evaluating videos...")
        print("=" * 60)

        for vid_idx, video_path in enumerate(all_videos):
            if video_path.name in ("gen.mp4", "output.mp4"):
                video_name = video_path.parent.name
            else:
                video_name = video_path.stem

            if video_name in resumed_results:
                clip_video_complete = (
                    lpips_clip is None
                    or video_name in all_clip_video_scores
                )

                if clip_video_complete:
                    skipped_count += 1
                    print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - SKIPPED (resumed)")
                    continue

                print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - resumed, completing missing metrics...", flush=True)

                if not clip_video_complete:
                    try:
                        print("    computing clip_video...", flush=True)
                        vr = VideoReader(str(video_path), ctx=decord_cpu(0))
                        effective_video_frames = _get_effective_video_frame_count(str(video_path), n_pose)
                        all_frames_rgb = [vr[i].asnumpy() for i in range(effective_video_frames)]
                        clip_score = lpips_clip.compute_clip_video_score(all_frames_rgb)
                        all_clip_video_scores[video_name] = clip_score
                        del all_frames_rgb, vr
                        print(f"    clip_video={clip_score:.4f}")
                    except (RuntimeError, Exception) as exc:
                        print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - clip_video ERROR: {exc}")

                save_partial_results(
                    output_dir,
                    all_results,
                    all_clip_video_scores,
                )
                continue

            print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name}", flush=True)

            try:
                eval_out = evaluate_video(
                    str(video_path), revisit_pairs, n_pose, dino_ext, boq_ext,
                    mvpr_ext=mvpr_ext,
                    keypoint_matcher=keypoint_matcher,
                    skip_pixel_metrics=args.skip_pixel_metrics,
                    lpips_clip=lpips_clip,
                    max_video_frames=n_pose,
                    max_eval_pairs=args.max_eval_pairs,
                )
            except (RuntimeError, Exception) as exc:
                print(f"    ERROR: {exc} — skipping this video")
                continue
            video_results = eval_out["pair_results"]
            all_results[video_name] = video_results
            if eval_out["clip_video"] is not None:
                all_clip_video_scores[video_name] = eval_out["clip_video"]
            all_revisit_info[video_name] = {
                "n_pose": n_pose,
                "revisit_groups": [{"name": g["name"], "num_pairs": len(g["pairs"])}
                                   for g in revisit_groups],
                "revisit_pairs": [{"frame_a": a, "frame_b": b}
                                  for a, b, _, _ in revisit_pairs],
            }

            # Incremental save
            save_partial_results(
                output_dir,
                all_results,
                all_clip_video_scores,
            )

            if not args.no_vis:
                visualize_pairs(str(video_path), revisit_pairs, video_results,
                                n_pose, output_dir, args.max_vis_pairs)

        if skipped_count:
            print(f"\n  Resumed: skipped {skipped_count} already-evaluated videos")

    else:
        # ------ Per-video pose mode: each video has its own pose file ------
        print("\n" + "=" * 60)
        print("Step 3: Evaluating videos (per-video pose)...")
        print("=" * 60)

        for vid_idx, video_path in enumerate(all_videos):
            if video_path.name in ("gen.mp4", "output.mp4"):
                video_name = video_path.parent.name
            else:
                video_name = video_path.stem

            video_parent = video_path.parent
            pose_path = video_parent / args.per_video_pose

            if video_name in resumed_results:
                clip_video_complete = (
                    lpips_clip is None
                    or video_name in all_clip_video_scores
                )

                resumed_pose_frames = None
                if pose_path.exists():
                    try:
                        _, resumed_pose_frames = extract_revisit_pairs(
                            str(pose_path),
                            angle_tolerance=args.angle_tolerance,
                            translation_tolerance=args.translation_tolerance,
                        )
                    except (RuntimeError, Exception) as exc:
                        print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - trim detection ERROR: {exc}")

                if clip_video_complete:
                    skipped_count += 1
                    print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - SKIPPED (resumed)")
                    continue

                print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - resumed, completing missing metrics...", flush=True)

                if not clip_video_complete:
                    try:
                        print("    computing clip_video...", flush=True)
                        vr = VideoReader(str(video_path), ctx=decord_cpu(0))
                        effective_video_frames = _get_effective_video_frame_count(str(video_path), resumed_pose_frames)
                        all_frames_rgb = [vr[i].asnumpy() for i in range(effective_video_frames)]
                        clip_score = lpips_clip.compute_clip_video_score(all_frames_rgb)
                        all_clip_video_scores[video_name] = clip_score
                        del all_frames_rgb, vr
                        print(f"    clip_video={clip_score:.4f}")
                    except (RuntimeError, Exception) as exc:
                        print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - clip_video ERROR: {exc}")

                save_partial_results(
                    output_dir,
                    all_results,
                    all_clip_video_scores,
                )
                continue

            if not pose_path.exists():
                print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - "
                      f"SKIP (no {args.per_video_pose})")
                continue

            print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name}", flush=True)

            # Detect revisit pairs for this video
            revisit_groups, n_pose = extract_revisit_pairs(
                str(pose_path),
                angle_tolerance=args.angle_tolerance,
                translation_tolerance=args.translation_tolerance,
            )
            revisit_pairs = flatten_revisit_groups(revisit_groups)
            n_pairs = len(revisit_pairs)
            total_revisit_count += n_pairs
            print(f"    Pose frames: {n_pose}, revisit pairs: {n_pairs}")

            if not revisit_pairs:
                print(f"    No revisit pairs, skipping")
                all_results[video_name] = []
                continue

            try:
                eval_out = evaluate_video(
                    str(video_path), revisit_pairs, n_pose, dino_ext, boq_ext,
                    mvpr_ext=mvpr_ext,
                    keypoint_matcher=keypoint_matcher,
                    skip_pixel_metrics=args.skip_pixel_metrics,
                    lpips_clip=lpips_clip,
                    max_video_frames=n_pose,
                    max_eval_pairs=args.max_eval_pairs,
                )
            except (RuntimeError, Exception) as exc:
                print(f"    ERROR: {exc} — skipping this video")
                continue
            video_results = eval_out["pair_results"]
            all_results[video_name] = video_results
            if eval_out["clip_video"] is not None:
                all_clip_video_scores[video_name] = eval_out["clip_video"]
            all_revisit_info[video_name] = {
                "n_pose": n_pose,
                "revisit_groups": [{"name": g["name"], "num_pairs": len(g["pairs"])}
                                   for g in revisit_groups],
                "revisit_pairs": [{"frame_a": a, "frame_b": b}
                                  for a, b, _, _ in revisit_pairs],
            }

            # Incremental save
            save_partial_results(
                output_dir,
                all_results,
                all_clip_video_scores,
            )

            if not args.no_vis:
                visualize_pairs(str(video_path), revisit_pairs, video_results,
                                n_pose, output_dir, args.max_vis_pairs)

        if skipped_count:
            print(f"\n  Resumed: skipped {skipped_count} already-evaluated videos")

    # =========================================================================
    # Step 4: Compute summary & correlations
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Computing summary & correlations...")
    print("=" * 60)

    global_summary, all_psnr = _collect_global_summary(
        all_results,
        all_clip_video_scores,
        total_revisit_count,
    )
    correlations = compute_correlations(all_results) if all_psnr else {}
    _print_summary(global_summary, correlations, total_revisit_count)

    # =========================================================================
    # Step 5: Save final results
    # =========================================================================
    final_output = {
        "config": {
            "video_dir": str(args.video_dir),
            "pose_json": args.pose_json or "per_video",
            "per_video_pose": args.per_video_pose,
            "angle_tolerance": args.angle_tolerance,
            "translation_tolerance": args.translation_tolerance,
            "num_videos_evaluated": len(all_results),
            "device": device,
        },
        "global_summary": global_summary,
        "correlations": correlations,
        "clip_video_scores": all_clip_video_scores if all_clip_video_scores else None,
        "per_video_revisit_info": all_revisit_info,
        "per_video_results": all_results,
    }

    results_path = output_dir / "results.json"
    _write_json_atomic(results_path, final_output)

    # Remove partial file
    partial_path = output_dir / "results_partial.json"
    if partial_path.exists():
        partial_path.unlink()

    print(f"\n  Results saved to: {results_path}")
    if not args.no_vis:
        print(f"  Visualizations saved to: {output_dir / 'visualizations'}")
    print("\nDone!")


if __name__ == "__main__":
    main()
