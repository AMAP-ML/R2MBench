"""
Normalized Memory Retention (NMR) Evaluation
=============================================
Computes NMR metric that normalizes revisit similarity against both
a negative baseline and a short-term upper bound:

  Similarity metrics (higher=better):
    NMR = (S_revisit - S_negative + eps) / (S_short - S_negative + eps)

  Distance metrics (lower=better, e.g. LPIPS):
    NMR = (D_negative - D_revisit) / (D_negative - D_short + eps)

Interpretation:
  NMR = 1  : revisit retains short-term consistency perfectly
  NMR = 0  : revisit is no better than random non-revisit pairs
  NMR < 0  : revisit is worse than random non-revisit pairs

Usage:
    python eval_revisit_nmr.py \
        --video_dir /path/to/videos \
        --pose_json /path/to/trajectory.json \
        --output_dir ./eval_nmr_results \
        --device cuda:0
"""

import os
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault(
    "TORCH_HOME",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "torch_hub_cache"),
)

import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import torch
from decord import VideoReader, cpu as decord_cpu
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

import math

from eval_revisit import (
    extract_revisit_pairs,
    flatten_revisit_groups,
    map_pose_to_video_frame,
    load_poses_from_json,
    _extract_frame_poses,
)
from metrics import (
    DINOv2FeatureExtractor,
    BoQFeatureExtractor,
    MutualVPRFeatureExtractor,
    KeypointMatcher,
    LpipsClipComputer,
    compute_psnr_ssim,
    compute_two_view_geometry,
)

# Shared pre-sampled pairs loader (unified pipeline)
from presampled_pairs import load_pairs_json, select_for_video

EPS = 1e-8


def _evaluate_frame_pairs(
    vr,
    pairs: List[Tuple[int, int]],
    dino_ext=None,
    boq_ext=None,
    mvpr_ext=None,
    keypoint_matcher=None,
    skip_pixel_metrics: bool = False,
    lpips_clip=None,
    label: str = "pairs",
) -> List[dict]:
    """Evaluate frame pairs with the selected visual metric families."""
    results = []
    for pair_idx, (vid_a, vid_b) in enumerate(pairs):
        img_a = vr[vid_a].asnumpy()
        img_b = vr[vid_b].asnumpy()

        entry = {
            "status": "ok",
            "video_frames": [vid_a, vid_b],
        }

        if not skip_pixel_metrics:
            entry.update(compute_psnr_ssim(img_a, img_b))

        if dino_ext is not None:
            entry["dino_similarity"] = round(dino_ext.similarity(img_a, img_b), 6)
        if boq_ext is not None:
            entry["boq_similarity"] = round(boq_ext.similarity(img_a, img_b), 6)
        if mvpr_ext is not None:
            entry["mvpr_similarity"] = round(mvpr_ext.similarity(img_a, img_b), 6)
        if lpips_clip is not None:
            entry["lpips"] = round(lpips_clip.compute_lpips(img_a, img_b), 6)
        if keypoint_matcher is not None:
            keypoint_result = keypoint_matcher.match(img_a, img_b)
            entry["num_keypoints_a"] = keypoint_result["num_keypoints_a"]
            entry["num_keypoints_b"] = keypoint_result["num_keypoints_b"]
            entry["num_matches"] = keypoint_result["num_matches"]
            entry["match_ratio"] = keypoint_result["match_ratio"]

            matched_keypoints_a = keypoint_result.get("matched_kp_a")
            matched_keypoints_b = keypoint_result.get("matched_kp_b")
            entry.update(
                compute_two_view_geometry(
                    matched_keypoints_a,
                    matched_keypoints_b,
                )
            )

        results.append(entry)

        if (pair_idx + 1) % 10 == 0:
            print(f"      {label}: {pair_idx + 1}/{len(pairs)} done", flush=True)

    return results

def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two yaw angles in degrees."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def _translation_distance(p1: dict, p2: dict) -> float:
    """Euclidean translation distance between two frame poses."""
    t1 = np.array([p1["tx"], p1["ty"], p1["tz"]], dtype=np.float64)
    t2 = np.array([p2["tx"], p2["ty"], p2["tz"]], dtype=np.float64)
    return float(np.linalg.norm(t1 - t2))


def _trajectory_diameter(frame_poses: Dict[int, dict]) -> float:
    """Maximum translation span of the trajectory, approximated by bbox diagonal."""
    if not frame_poses:
        return 0.0

    pts = np.array(
        [[p["tx"], p["ty"], p["tz"]] for p in frame_poses.values()],
        dtype=np.float64,
    )
    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)
    return float(np.linalg.norm(xyz_max - xyz_min))


def _build_exclusion_frames(
    revisit_video_pairs: List[Tuple[int, int]],
    n_video: int,
    exclusion_radius: int,
) -> Set[int]:
    """Build frame set around all revisit frames to exclude from baseline sampling."""
    excluded = set()

    for a, b in revisit_video_pairs:
        for rf in (a, b):
            lo = max(0, rf - exclusion_radius)
            hi = min(n_video - 1, rf + exclusion_radius)
            for f in range(lo, hi + 1):
                excluded.add(f)

    return excluded


def _make_temporal_bins(
    frames: List[int],
    n_video: int,
    num_temporal_bins: int,
) -> List[List[int]]:
    """Group frame indices into temporal bins."""
    num_temporal_bins = max(1, int(num_temporal_bins))
    bin_size = max(1, math.ceil(n_video / num_temporal_bins))

    bins = [[] for _ in range(num_temporal_bins)]
    for f in frames:
        b = min(f // bin_size, num_temporal_bins - 1)
        bins[b].append(f)

    return bins


def _get_bin_index(frame_idx: int, n_video: int, num_temporal_bins: int) -> int:
    num_temporal_bins = max(1, int(num_temporal_bins))
    bin_size = max(1, math.ceil(n_video / num_temporal_bins))
    return min(frame_idx // bin_size, num_temporal_bins - 1)


def _candidate_frames_from_expanded_bins(
    temporal_bins: List[List[int]],
    target_bin: int,
    max_expand: Optional[int] = None,
) -> List[int]:
    """Return candidates by progressively expanding from target temporal bin."""
    num_bins = len(temporal_bins)
    if max_expand is None:
        max_expand = num_bins

    for radius in range(max_expand + 1):
        lo = max(0, target_bin - radius)
        hi = min(num_bins, target_bin + radius + 1)

        candidates = []
        for b in range(lo, hi):
            candidates.extend(temporal_bins[b])

        if len(candidates) >= 2:
            return candidates

    # final fallback: all available frames
    candidates = []
    for bin_frames in temporal_bins:
        candidates.extend(bin_frames)
    return candidates


def _sample_gap_matched_temporal_baselines(
    revisit_video_pairs: List[Tuple[int, int]],
    frame_poses: Dict[int, dict],
    n_video: int,
    num_temporal_bins: int = 10,
    exclusion_radius: int = 10,
    gap_tolerance_ratio: float = 0.3,
    min_gap_tolerance: int = 3,
    max_attempts_per_pair: int = 500,
    allow_duplicate: bool = False,
    seed: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], dict]:
    """Sample gap-matched temporal baseline pairs for strict pair-level NMR.

    For each revisit pair (i, j) with i < j, sample one baseline pair
    (u, u+Δ) such that:
      1. |Δ - (j-i)| <= delta_t  (temporal gap matching, forward-only)
      2. NotNearReturn: if |u - i_k| <= r then |(u+Δ) - j_k| > r
      3. u is sampled from the same temporal bin as the revisit anchor i
      4. baseline pairs are unique by default

    We intentionally do NOT enforce a spatial-far constraint, so slow or
    low-motion generations receive a high baseline similarity and therefore
    a lower revisit advantage. The baseline definition is purely temporal.

    Poses are used only to ensure frame validity and for diagnostic
    statistics, not for baseline filtering.

    Args:
        revisit_video_pairs:
            List of (frame_a, frame_b) in video-frame space.
            The earlier frame is treated as the anchor.
        frame_poses:
            Dict mapping frame index to {"yaw", "tx", "ty", "tz"}.
            Used only for frame validity checks and diagnostic stats.
        n_video:
            Number of video frames.
        num_temporal_bins:
            Number of bins for anchor-time stratification.
        exclusion_radius:
            Radius r for NotNearReturn check (frames).
        gap_tolerance_ratio:
            Fraction of revisit gap used as tolerance window half-width.
        min_gap_tolerance:
            Minimum absolute tolerance (frames) for gap matching.
        max_attempts_per_pair:
            Maximum random attempts for each revisit pair.
        allow_duplicate:
            Whether to allow repeated baseline pairs.
        seed:
            Optional local random seed.

    Returns:
        baseline_pairs:
            List of sampled (u, v) pairs with u < v (forward-only).
        stats:
            Diagnostic statistics.
    """
    rng = random.Random(seed)

    if not revisit_video_pairs or n_video <= 1:
        return [], {
            "num_revisit": len(revisit_video_pairs),
            "num_baseline": 0,
            "matched_rate": 0.0,
            "reason": "empty_input",
        }

    # Keep only revisit pairs with valid frame indices and valid poses.
    valid_revisit = []
    for a, b in revisit_video_pairs:
        if a == b:
            continue
        if not (0 <= a < n_video and 0 <= b < n_video):
            continue
        if a not in frame_poses or b not in frame_poses:
            continue
        anchor, ret = (a, b) if a < b else (b, a)
        valid_revisit.append((anchor, ret))

    if not valid_revisit:
        return [], {
            "num_revisit": len(revisit_video_pairs),
            "num_valid_revisit": 0,
            "num_baseline": 0,
            "matched_rate": 0.0,
            "reason": "no_valid_revisit",
        }

    # Build revisit frame set (u/v cannot be a revisit frame itself).
    revisit_frame_set: Set[int] = set()
    for anchor, ret in valid_revisit:
        revisit_frame_set.add(anchor)
        revisit_frame_set.add(ret)

    # Available frames: all frames with pose data, excluding revisit frames.
    available_frames = [
        f for f in range(n_video)
        if f in frame_poses and f not in revisit_frame_set
    ]

    print(f"    [DEBUG] n_video={n_video}, valid_revisit={len(valid_revisit)}, "
          f"revisit_frames={len(revisit_frame_set)}, available={len(available_frames)}, "
          f"exclusion_radius={exclusion_radius}")

    if len(available_frames) < 2:
        return [], {
            "num_revisit": len(revisit_video_pairs),
            "num_valid_revisit": len(valid_revisit),
            "num_baseline": 0,
            "matched_rate": 0.0,
            "reason": "too_few_available_frames",
        }

    temporal_bins = _make_temporal_bins(
        available_frames,
        n_video=n_video,
        num_temporal_bins=num_temporal_bins,
    )

    used_pairs: Set[Tuple[int, int]] = set()
    baseline_pairs: List[Tuple[int, int]] = []

    num_no_candidate = 0
    num_attempt_failed = 0
    gap_errors = []
    trans_dists = []
    yaw_diffs = []

    def _check_not_near_return(u: int, v: int) -> bool:
        """Strict NotNearReturn (forward-only, u < v guaranteed):
        if u is near any revisit anchor i_k, then v must be far from
        the corresponding return j_k. Neither frame can be a revisit frame."""
        if u in revisit_frame_set or v in revisit_frame_set:
            return False
        for anchor_k, ret_k in valid_revisit:
            if abs(u - anchor_k) <= exclusion_radius and abs(v - ret_k) <= exclusion_radius:
                return False
        return True

    debug_counts = {"out_of_range": 0, "no_pose": 0, "near_return": 0,
                    "duplicate": 0, "candidate_accepted": 0, "pair_selected": 0}

    for pair_idx, (anchor, ret) in enumerate(valid_revisit):
        rev_gap = ret - anchor
        if rev_gap <= 0:
            continue

        delta_t = max(10, int(round(gap_tolerance_ratio * rev_gap)))

        gap_low = max(1, rev_gap - delta_t)
        gap_high = min(n_video - 1, rev_gap + delta_t)

        target_bin = _get_bin_index(
            anchor,
            n_video=n_video,
            num_temporal_bins=num_temporal_bins,
        )

        candidate_anchors = _candidate_frames_from_expanded_bins(
            temporal_bins,
            target_bin=target_bin,
        )

        # Forward-only: filter anchors that cannot reach gap_high forward.
        candidate_anchors = [
            f for f in candidate_anchors
            if f + gap_high < n_video
        ]
        # Fallback to gap_low if too strict.
        if len(candidate_anchors) < 1:
            candidate_anchors = [
                f for f in _candidate_frames_from_expanded_bins(temporal_bins, target_bin)
                if f + gap_low < n_video
            ]

        if len(candidate_anchors) < 1:
            num_no_candidate += 1
            continue

        best_pair = None
        best_score = float("inf")

        # Strict gap-matched, forward-only search. If no valid baseline
        # is found within the gap window, this revisit pair is left unmatched.
        for _ in range(max_attempts_per_pair):
            u = rng.choice(candidate_anchors)
            gap = rng.randint(gap_low, gap_high)

            # Forward-only: v = u + gap, always u < v.
            v = u + gap
            if v >= n_video:
                debug_counts["out_of_range"] += 1
                continue
            if v not in frame_poses:
                debug_counts["no_pose"] += 1
                continue
            if not _check_not_near_return(u, v):
                debug_counts["near_return"] += 1
                continue

            pair = (u, v)  # u < v guaranteed by forward-only
            if not allow_duplicate and pair in used_pairs:
                debug_counts["duplicate"] += 1
                continue

            debug_counts["candidate_accepted"] += 1

            gap_error = abs(gap - rev_gap)

            # Prefer better gap match.
            score = gap_error

            if score < best_score:
                best_score = score
                best_pair = pair

                # Perfect gap match: early stop
                if gap_error == 0:
                    break

        if best_pair is None:
            num_attempt_failed += 1
            continue

        debug_counts["pair_selected"] += 1
        baseline_pairs.append(best_pair)
        if not allow_duplicate:
            used_pairs.add(best_pair)

        gap_errors.append(abs((best_pair[1] - best_pair[0]) - rev_gap))

        # Compute pose distances for diagnostic only (not used for filtering).
        diag_trans = _translation_distance(frame_poses[best_pair[0]], frame_poses[best_pair[1]])
        trans_dists.append(diag_trans)
        yaw_u = frame_poses[best_pair[0]].get("yaw", None)
        yaw_v = frame_poses[best_pair[1]].get("yaw", None)
        if yaw_u is not None and yaw_v is not None:
            yaw_diffs.append(_angle_diff_deg(float(yaw_u), float(yaw_v)))

    print(f"    [DEBUG] Sampling stats: {debug_counts}, "
          f"no_candidate={num_no_candidate}, attempt_failed={num_attempt_failed}")

    # Fallback: if strict gap-matched sampling yields too few baseline pairs
    # (e.g. revisit_loop where only start/end are revisit frames), relax the
    # gap tolerance to 0.5 * rev_gap and re-sample. The target gap center
    # stays at rev_gap so the baseline gap distribution remains consistent
    # with revisit pairs. This ensures at least MIN_BASELINE_PAIRS for
    # statistical reliability.
    MIN_BASELINE_PAIRS = 20
    num_fallback = 0
    if len(baseline_pairs) < MIN_BASELINE_PAIRS and len(available_frames) >= 2:
        needed = MIN_BASELINE_PAIRS - len(baseline_pairs)
        print(f"    [DEBUG] Baseline insufficient ({len(baseline_pairs)}<{MIN_BASELINE_PAIRS}), "
              f"relaxing gap tolerance to 0.5*rev_gap for fallback sampling")

        fb_temporal_bins = _make_temporal_bins(
            available_frames, n_video=n_video, num_temporal_bins=num_temporal_bins,
        )

        for anchor, ret in valid_revisit:
            if num_fallback >= needed:
                break
            rev_gap = ret - anchor
            if rev_gap <= 0:
                continue

            # Same target gap as revisit; tolerance relaxed to 50% of rev_gap.
            relaxed_delta_t = max(min_gap_tolerance, int(round(0.5 * rev_gap)))
            fb_gap_low = max(1, rev_gap - relaxed_delta_t)
            fb_gap_high = min(n_video - 1, rev_gap + relaxed_delta_t)

            target_bin = _get_bin_index(anchor, n_video=n_video, num_temporal_bins=num_temporal_bins)
            fb_candidates = _candidate_frames_from_expanded_bins(fb_temporal_bins, target_bin)
            fb_candidates = [f for f in fb_candidates if f + fb_gap_high < n_video]
            if not fb_candidates:
                fb_candidates = [f for f in _candidate_frames_from_expanded_bins(fb_temporal_bins, target_bin)
                                 if f + fb_gap_low < n_video]
            if not fb_candidates:
                continue

            for _ in range(max_attempts_per_pair):
                if num_fallback >= needed:
                    break
                u = rng.choice(fb_candidates)
                gap = rng.randint(fb_gap_low, fb_gap_high)
                v = u + gap
                if v >= n_video:
                    continue
                if v not in frame_poses:
                    continue
                pair = (u, v)
                if pair in used_pairs:
                    continue
                if not _check_not_near_return(u, v):
                    continue
                baseline_pairs.append(pair)
                used_pairs.add(pair)
                num_fallback += 1

                gap_errors.append(abs(gap - rev_gap))
                diag_trans = _translation_distance(frame_poses[u], frame_poses[v])
                trans_dists.append(diag_trans)
                yaw_u_fb = frame_poses[u].get("yaw", None)
                yaw_v_fb = frame_poses[v].get("yaw", None)
                if yaw_u_fb is not None and yaw_v_fb is not None:
                    yaw_diffs.append(_angle_diff_deg(float(yaw_u_fb), float(yaw_v_fb)))

        print(f"    [DEBUG] Fallback relaxed-gap sampling: added {num_fallback} pairs "
              f"(total now {len(baseline_pairs)}, target was {MIN_BASELINE_PAIRS})")

    stats = {
        "num_revisit": len(revisit_video_pairs),
        "num_valid_revisit": len(valid_revisit),
        "num_baseline": len(baseline_pairs),
        "num_fallback_baseline": num_fallback,
        "matched_rate": (
            float(len(baseline_pairs) / len(valid_revisit))
            if valid_revisit else 0.0
        ),
        "exclusion_radius": int(exclusion_radius),
        "num_temporal_bins": int(num_temporal_bins),
        "max_attempts_per_pair": int(max_attempts_per_pair),
        "num_no_candidate": int(num_no_candidate),
        "num_attempt_failed": int(num_attempt_failed),
        "mean_gap_error": float(np.mean(gap_errors)) if gap_errors else None,
        "median_gap_error": float(np.median(gap_errors)) if gap_errors else None,
        "mean_negative_translation": float(np.mean(trans_dists)) if trans_dists else None,
        "median_negative_translation": float(np.median(trans_dists)) if trans_dists else None,
        "mean_negative_yaw": float(np.mean(yaw_diffs)) if yaw_diffs else None,
        "median_negative_yaw": float(np.median(yaw_diffs)) if yaw_diffs else None,
    }

    return baseline_pairs, stats

def _sample_short_temporal_pairs(
    frame_poses: Dict[int, dict],
    n_video: int,
    count: int = 20,
    min_gap: int = 2,
    max_gap: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Tuple[int, int]], dict]:
    """Sample short-range temporal pairs as local consistency reference.

    P_short = {(a, b) : min_gap <= b - a <= max_gap}

    Short pairs measure the model's local temporal consistency within
    the same generated video. They are NOT pose-based; the definition is
    purely temporal. Poses are used only for frame validity checks.

    Args:
        frame_poses: Dict mapping frame index to pose data.
            Used only for frame validity checks.
        n_video: Number of video frames.
        count: How many short pairs to sample.
        min_gap: Minimum temporal gap (frames). Default 2.
        max_gap: Maximum temporal gap. If None, max(6, floor(0.02 * n_video)).
        seed: Optional random seed.

    Returns:
        Tuple of (pairs, stats) where pairs is a list of (a, b) tuples
        with a < b (forward-only), and stats contains sampling diagnostics.
    """
    rng = random.Random(seed)

    empty_stats = {
        "num_short": 0, "target_count": count,
        "matched_rate": 0.0, "min_gap": min_gap, "max_gap": max_gap or 0,
    }

    if n_video <= 1:
        return [], empty_stats

    if max_gap is None:
        max_gap = max(6, int(0.02 * n_video))

    min_gap = max(1, int(min_gap))
    max_gap = max(min_gap, int(max_gap))

    # Valid frames: those with pose data. Use a set for O(1) lookup.
    valid_frames = [f for f in range(n_video) if f in frame_poses]
    valid_set = set(valid_frames)
    if len(valid_frames) < 2:
        return [], empty_stats

    # Anchors only need min_gap headroom; per-anchor gap_high handles the rest.
    candidate_anchors = [f for f in valid_frames if f + min_gap < n_video]
    if len(candidate_anchors) < 1:
        return [], empty_stats

    pairs: List[Tuple[int, int]] = []
    used: Set[Tuple[int, int]] = set()

    # Stratify anchors across valid range.
    anchor_indices = np.linspace(
        0, len(candidate_anchors) - 1, num=min(count, len(candidate_anchors))
    )
    anchor_list = [candidate_anchors[int(round(x))] for x in anchor_indices]

    attempts_per_anchor = max(10, 5000 // max(1, len(anchor_list)))

    for anchor in anchor_list:
        # Dynamically cap gap so we never sample beyond video end.
        gap_high = min(max_gap, n_video - 1 - anchor)
        if gap_high < min_gap:
            continue

        for _ in range(attempts_per_anchor):
            gap = rng.randint(min_gap, gap_high)
            b = anchor + gap

            if b not in valid_set:
                continue

            pair = (anchor, b)  # forward-only, anchor < b guaranteed
            if pair in used:
                continue

            pairs.append(pair)
            used.add(pair)
            break  # One pair per anchor is sufficient

    short_stats = {
        "num_short": len(pairs),
        "target_count": count,
        "matched_rate": len(pairs) / max(1, count),
        "min_gap": min_gap,
        "max_gap": max_gap,
    }
    return pairs, short_stats


MIN_DYNAMIC_RANGE = 1e-5


def _compute_nmr(revisit_agg: dict, baseline_agg: dict, short_agg: dict) -> dict:
    """Compute Normalized Memory Retention for each metric.

    For similarity metrics (higher=better):
        NMR = (S_revisit - S_negative) / (S_short - S_negative)

    For distance metrics (lower=better, e.g. LPIPS):
        NMR = (D_negative - D_revisit) / (D_negative - D_short)

    If the dynamic range between short and negative is below MIN_DYNAMIC_RANGE,
    NMR is set to 0.0 (model has insufficient temporal contrast).
    """
    lower_is_better = {"lpips"}
    # num_matches is an absolute count heavily dependent on texture/brightness,
    # not suitable for NMR normalization. Report raw values only.
    skip_nmr = {"num_matches"}
    nmr = {}
    nmr_valid = {}

    for key in revisit_agg:
        if key in skip_nmr:
            continue
        if key not in baseline_agg or key not in short_agg:
            continue

        revisit_mean = revisit_agg[key]["mean"]
        negative_mean = baseline_agg[key]["mean"]
        short_mean = short_agg[key]["mean"]

        if key in lower_is_better:
            dynamic_range = negative_mean - short_mean
            numerator = negative_mean - revisit_mean
        else:
            dynamic_range = short_mean - negative_mean
            numerator = revisit_mean - negative_mean

        if dynamic_range <= MIN_DYNAMIC_RANGE:
            nmr[key] = 0.0
            nmr_valid[key] = False
        else:
            nmr[key] = round(float(numerator / dynamic_range), 6)
            nmr_valid[key] = True

    return nmr


def _aggregate_metrics(results: List[dict]) -> dict:
    """Compute mean of each metric across results."""
    metric_keys = [
        "psnr", "ssim", "dino_similarity", "boq_similarity", "mvpr_similarity",
        "lpips", "num_matches", "match_ratio", "ransac_inlier_ratio",
    ]
    aggregated = {}
    for key in metric_keys:
        values = [r[key] for r in results if r.get("status") == "ok" and key in r]
        if values:
            aggregated[key] = {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6),
                "median": round(float(np.median(values)), 6),
                "count": len(values),
            }
    return aggregated


def _compute_global_summary(all_per_video: dict):
    """Compute global summary: average revisit/baseline/short first, then compute NMR.

    NMR = (mean_revisit - mean_baseline) / (mean_short - mean_baseline)
    This avoids bias from averaging per-video NMR values with different denominators.
    """
    metric_keys = [
        "psnr", "ssim", "dino_similarity", "boq_similarity", "mvpr_similarity",
        "lpips", "num_matches", "match_ratio", "ransac_inlier_ratio",
    ]

    revisit_totals = {k: [] for k in metric_keys}
    baseline_totals = {k: [] for k in metric_keys}
    short_totals = {k: [] for k in metric_keys}

    for video_result in all_per_video.values():
        if not isinstance(video_result, dict):
            continue
        revisit = video_result.get("revisit", {})
        baseline = video_result.get("baseline", {})
        short = video_result.get("short", {})

        for key in metric_keys:
            if key in revisit and isinstance(revisit[key], dict):
                revisit_totals[key].append(revisit[key]["mean"])
            if key in baseline and isinstance(baseline[key], dict):
                baseline_totals[key].append(baseline[key]["mean"])
            if key in short and isinstance(short[key], dict):
                short_totals[key].append(short[key]["mean"])

    global_revisit = {}
    global_baseline = {}
    global_short = {}
    global_nmr = {}

    for key in metric_keys:
        if revisit_totals[key]:
            global_revisit[key] = round(float(np.mean(revisit_totals[key])), 6)
        if baseline_totals[key]:
            global_baseline[key] = round(float(np.mean(baseline_totals[key])), 6)
        if short_totals[key]:
            global_short[key] = round(float(np.mean(short_totals[key])), 6)

        # Compute NMR from averaged values: (R - B) / (S - B)
        if revisit_totals[key] and baseline_totals[key] and short_totals[key]:
            mean_rev = float(np.mean(revisit_totals[key]))
            mean_base = float(np.mean(baseline_totals[key]))
            mean_short = float(np.mean(short_totals[key]))
            denominator = mean_short - mean_base
            if abs(denominator) > 1e-8:
                global_nmr[key] = round((mean_rev - mean_base) / denominator, 6)

    return global_nmr, global_revisit, global_baseline, global_short


def _save_results(output_dir: Path, all_per_video: dict):
    """Save partial results atomically."""
    payload = {"per_video": all_per_video}
    temp_path = output_dir / "results_partial.json.tmp"
    output_path = output_dir / "results_partial.json"
    with open(temp_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Normalized Memory Retention (NMR) Evaluation"
    )
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files")
    parser.add_argument("--pose_json", type=str, default=None,
                        help="Global trajectory pose JSON")
    parser.add_argument("--per_video_pose", type=str, default=None,
                        help="Per-video pose filename")
    parser.add_argument("--output_dir", type=str, default="./eval_nmr_results",
                        help="Output directory for results")
    parser.add_argument("--angle_tolerance", type=float, default=None,
                        help="Max yaw difference in degrees (default: adaptive)")
    parser.add_argument("--translation_tolerance", type=float, default=None,
                        help="Max translation difference in meters (default: adaptive)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--num_video_shards", type=int, default=1)
    parser.add_argument("--video_shard_id", type=int, default=0)
    parser.add_argument("--skip_pixel_metrics", action="store_true")
    parser.add_argument("--only_keypoints", action="store_true")
    parser.add_argument("--enable_lpips", action="store_true")
    parser.add_argument("--clip_checkpoint", type=str,
                        default=os.path.expanduser("~/.cache/clip/ViT-B-32.pt"))
    parser.add_argument("--max_eval_pairs", type=int, default=200)
    parser.add_argument("--baseline_multiplier", type=float, default=1.0)
    parser.add_argument("--short_multiplier", type=float, default=1.0,
                        help="Number of short pairs = revisit_pairs * multiplier")
    parser.add_argument("--short_max_gap", type=int, default=3,
                        help="Max frame gap for short-term pairs")
    parser.add_argument("--min_gap_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pairs_json", type=str, default=None,
                        help="Pre-sampled shared pairs JSON (revisit/baseline/short "
                             "in video-frame space). When set, these pairs are used "
                             "instead of per-video internal sampling so all eval "
                             "types score identical pairs.")
    parser.add_argument("--families", type=str, default="all",
                        help="Comma-separated subset of visual metric families to "
                             "compute: appearance (PSNR/SSIM/LPIPS), scene_identity "
                             "(DINOv2/BoQ/MutualVPR), geometric (SuperPoint+LightGlue). "
                             "Default 'all'. Only the models needed by the selected "
                             "families are loaded.")
    args = parser.parse_args()

    if not args.pose_json and not args.per_video_pose:
        parser.error("Either --pose_json or --per_video_pose must be specified")
    if args.num_video_shards < 1:
        parser.error("--num_video_shards must be >= 1")
    if args.video_shard_id < 0 or args.video_shard_id >= args.num_video_shards:
        parser.error("--video_shard_id must be in [0, num_video_shards)")

    if args.only_keypoints:
        args.skip_pixel_metrics = True

    # Parse the visual metric-family selector. The three families map onto the
    # optional extractors of _evaluate_frame_pairs:
    #   appearance     -> PSNR/SSIM (+ LPIPS when --enable_lpips)
    #   scene_identity -> DINOv2 + BoQ + MutualVPR
    #   geometric      -> SuperPoint + LightGlue (+ two-view geometry)
    _VALID_FAMILIES = {"appearance", "scene_identity", "geometric"}
    _fam_raw = [t.strip().lower() for t in args.families.split(",") if t.strip()]
    if not _fam_raw or "all" in _fam_raw:
        selected_families = set(_VALID_FAMILIES)
    else:
        selected_families = set()
        for tok in _fam_raw:
            if tok not in _VALID_FAMILIES:
                parser.error(
                    f"--families: unknown family '{tok}'. "
                    f"Valid: appearance, scene_identity, geometric, all."
                )
            selected_families.add(tok)

    # Legacy flag reconciliation.
    if args.only_keypoints:
        # Backward-compatible shortcut: keypoints/geometry only.
        selected_families = {"geometric"}

    want_appearance = "appearance" in selected_families
    want_scene = "scene_identity" in selected_families
    want_geometric = "geometric" in selected_families

    # Appearance pixel metrics (PSNR/SSIM) and LPIPS are only produced when the
    # appearance family is selected.
    if not want_appearance:
        args.skip_pixel_metrics = True
        args.enable_lpips = False

    print(f"  Selected metric families: {sorted(selected_families)}")

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    resumed_results = {}
    if args.resume:
        for fname in ["results.json", "results_partial.json"]:
            fpath = output_dir / fname
            if fpath.exists():
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "per_video" in data:
                        resumed_results.update(data["per_video"])
                        print(f"  Resumed {len(resumed_results)} videos from {fname}")
                except (json.JSONDecodeError, OSError) as exc:
                    print(f"  Resume warning: {fname}: {exc}")

    # Load pre-sampled shared pairs (unified pipeline). When provided, all eval
    # types consume identical revisit/baseline/short pairs instead of each
    # sampling independently per video.
    presampled_pairs = None
    if args.pairs_json:
        presampled_pairs = load_pairs_json(args.pairs_json)
        print(f"  Using pre-sampled pairs from {args.pairs_json}: "
              f"revisit={len(presampled_pairs['revisit'])}, "
              f"baseline={len(presampled_pairs['baseline'])}, "
              f"short={len(presampled_pairs['short'])}")

    # Load models
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    device = args.device if torch.cuda.is_available() else "cpu"

    # Scene Identity family: DINOv2 + BoQ + MutualVPR global descriptors.
    if want_scene:
        dino_ext = DINOv2FeatureExtractor(device=device)
        boq_ext = BoQFeatureExtractor(device=device)
        mvpr_ext = MutualVPRFeatureExtractor(device=device)
    else:
        dino_ext = None
        boq_ext = None
        mvpr_ext = None

    # Local Geometric Correspondence family: SuperPoint + LightGlue.
    keypoint_matcher = KeypointMatcher(device=device) if want_geometric else None

    # Appearance Fidelity family: LPIPS (PSNR/SSIM are computed inline when the
    # appearance family is selected, i.e. when skip_pixel_metrics is False).
    lpips_clip = None
    if args.enable_lpips:
        lpips_clip = LpipsClipComputer(clip_checkpoint_path=args.clip_checkpoint, device=device)

    # Discover videos
    print("\n" + "=" * 60)
    print("Discovering videos...")
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

    num_discovered_videos = len(all_videos)
    all_videos = all_videos[args.video_shard_id::args.num_video_shards]

    print(f"  Found {num_discovered_videos} videos")
    if args.num_video_shards > 1:
        print(f"  Video shard {args.video_shard_id}/{args.num_video_shards}: {len(all_videos)} videos")

    # Evaluate
    per_video_pose_mode = args.per_video_pose is not None
    all_per_video = dict(resumed_results)

    def _get_pose_and_pairs(pose_json_path):
        revisit_groups, n_pose = extract_revisit_pairs(
            pose_json_path,
            angle_tolerance=args.angle_tolerance,
            translation_tolerance=args.translation_tolerance,
        )
        flat_pairs = flatten_revisit_groups(revisit_groups)
        # Also load frame poses for matched hard negative sampling
        pose_data, sorted_keys, _ = load_poses_from_json(pose_json_path)
        frame_poses = _extract_frame_poses(pose_data, sorted_keys)
        return flat_pairs, n_pose, frame_poses

    global_pairs = None
    global_n_pose = None
    global_frame_poses = None
    if not per_video_pose_mode:
        print("\n" + "=" * 60)
        print("Detecting revisit pairs (global pose)...")
        print("=" * 60)
        global_pairs, global_n_pose, global_frame_poses = _get_pose_and_pairs(args.pose_json)
        print(f"  Poses: {global_n_pose}, Revisit pairs: {len(global_pairs)}")
        if not global_pairs and presampled_pairs is None:
            print("  ERROR: No revisit pairs detected!")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Evaluating videos (NMR)...")
    print("=" * 60)

    for vid_idx, video_path in enumerate(all_videos):
        if video_path.name in ("gen.mp4", "output.mp4"):
            video_name = video_path.parent.name
        else:
            video_name = video_path.stem

        if video_name in resumed_results:
            print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - SKIPPED (resumed)")
            continue

        if per_video_pose_mode:
            pose_path = video_path.parent / args.per_video_pose
            if not pose_path.exists():
                print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - SKIPPED (no pose)")
                continue
            revisit_pairs, n_pose, frame_poses = _get_pose_and_pairs(str(pose_path))
        else:
            revisit_pairs = global_pairs
            n_pose = global_n_pose
            frame_poses = global_frame_poses

        if not revisit_pairs and presampled_pairs is None:
            print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - SKIPPED (no revisit pairs)")
            continue

        print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} ({len(revisit_pairs)} revisit pairs)", flush=True)

        try:
            vr = VideoReader(str(video_path), ctx=decord_cpu(0))
        except Exception as exc:
            print(f"    ERROR opening video: {exc}")
            continue

        n_video = len(vr)
        effective_frames = min(n_video, n_pose) if n_pose else n_video

        if presampled_pairs is not None:
            # Unified pipeline: use shared pre-sampled pairs (video-frame space).
            revisit_video_pairs, baseline_pairs, short_pairs = select_for_video(
                presampled_pairs, n_video, max_per_type=args.max_eval_pairs
            )
            if not revisit_video_pairs:
                print(f"    No valid pre-sampled revisit pairs for this video")
                del vr
                continue
            short_stats = {
                "target_count": len(short_pairs),
                "matched_rate": 1.0,
                "min_gap": 2,
                "max_gap": args.short_max_gap,
            }
            print(f"    [pre-sampled] Revisit: {len(revisit_video_pairs)}, "
                  f"Baseline: {len(baseline_pairs)}, Short: {len(short_pairs)}")
        else:
            # Filter revisit pairs by minimum frame gap
            min_gap = int(effective_frames * args.min_gap_ratio)
            filtered_revisit = []
            for fa, fb, dpos, drot in revisit_pairs:
                vid_a = map_pose_to_video_frame(fa, n_pose, n_video)
                vid_b = map_pose_to_video_frame(fb, n_pose, n_video)
                if vid_a >= effective_frames or vid_b >= effective_frames:
                    continue
                if abs(vid_b - vid_a) >= min_gap:
                    filtered_revisit.append((fa, fb, dpos, drot))

            if len(filtered_revisit) < len(revisit_pairs):
                print(f"    Gap filter (>={min_gap} frames): {len(filtered_revisit)}/{len(revisit_pairs)} kept")

            if not filtered_revisit:
                print(f"    No valid revisit video frame pairs")
                continue

            # Sample revisit pairs if too many
            sampled_revisit = filtered_revisit
            if args.max_eval_pairs and len(filtered_revisit) > args.max_eval_pairs:
                sampled_revisit = random.sample(filtered_revisit, args.max_eval_pairs)
                print(f"    Sampled {args.max_eval_pairs}/{len(filtered_revisit)} revisit pairs")

            # Convert to video-frame space
            revisit_video_pairs = []
            for fa, fb, _, _ in sampled_revisit:
                vid_a = map_pose_to_video_frame(fa, n_pose, n_video)
                vid_b = map_pose_to_video_frame(fb, n_pose, n_video)
                revisit_video_pairs.append((vid_a, vid_b))

            # Collect all revisit frame indices for exclusion from short pairs
            revisit_frame_set = set()
            for va, vb in revisit_video_pairs:
                revisit_frame_set.add(va)
                revisit_frame_set.add(vb)

            # Sample matched hard negative baseline pairs
            # Each revisit pair gets one temporal-gap-matched baseline with
            # NotNearReturn constraint. No spatial-far filtering.
            # Adaptive: for short videos, use smaller exclusion radius and fewer bins
            # to ensure enough baseline candidates are available.
            if effective_frames <= 200:
                exclusion_radius = max(5, effective_frames // 50)
                num_temporal_bins = max(3, effective_frames // 40)
                gap_tolerance_ratio=0.3
            else:
                exclusion_radius = max(10, effective_frames // 30)
                num_temporal_bins = 10
                gap_tolerance_ratio=0.3
            baseline_pairs, baseline_stats = _sample_gap_matched_temporal_baselines(
                revisit_video_pairs, frame_poses, effective_frames,
                num_temporal_bins=num_temporal_bins,
                exclusion_radius=exclusion_radius,
                gap_tolerance_ratio=gap_tolerance_ratio,
                seed=args.seed,
            )
            matched_rate = baseline_stats.get("matched_rate", 0.0)
            mean_neg_trans = baseline_stats.get("mean_negative_translation")
            mean_neg_yaw = baseline_stats.get("mean_negative_yaw")
            diag_str = ""
            if mean_neg_trans is not None:
                diag_str += f", mean_neg_trans={mean_neg_trans:.3f}m"
            if mean_neg_yaw is not None:
                diag_str += f", mean_neg_yaw={mean_neg_yaw:.1f}°"
            print(f"    Matched baselines: {len(baseline_pairs)}/{len(revisit_video_pairs)} "
                  f"(rate={matched_rate:.2%}{diag_str})")
            # Sample short-range temporal pairs (pure frame-distance, no pose filtering)
            short_count = min(100, max(20, int(len(revisit_video_pairs) * args.short_multiplier)))
            short_pairs, short_stats = _sample_short_temporal_pairs(
                frame_poses, effective_frames,
                count=short_count,
                min_gap=2,
                seed=args.seed,
            )

            print(f"    Revisit: {len(revisit_video_pairs)}, Baseline: {len(baseline_pairs)}, "
                  f"Short: {len(short_pairs)}/{short_stats['target_count']} "
                  f"(rate={short_stats['matched_rate']:.2%}, "
                  f"gap=[{short_stats['min_gap']},{short_stats['max_gap']}])")

        # Evaluate all three types of pairs
        eval_kwargs = dict(
            dino_ext=dino_ext, boq_ext=boq_ext, mvpr_ext=mvpr_ext,
            keypoint_matcher=keypoint_matcher,
            skip_pixel_metrics=args.skip_pixel_metrics,
            lpips_clip=lpips_clip,
        )

        print(f"    Evaluating revisit pairs...", flush=True)
        revisit_results = _evaluate_frame_pairs(vr, revisit_video_pairs, label="revisit", **eval_kwargs)

        print(f"    Evaluating baseline pairs...", flush=True)
        baseline_results = _evaluate_frame_pairs(vr, baseline_pairs, label="baseline", **eval_kwargs)

        print(f"    Evaluating short pairs...", flush=True)
        short_results = _evaluate_frame_pairs(vr, short_pairs, label="short", **eval_kwargs)

        del vr

        # Aggregate
        revisit_agg = _aggregate_metrics(revisit_results)
        baseline_agg = _aggregate_metrics(baseline_results)
        short_agg = _aggregate_metrics(short_results)

        # Compute NMR
        nmr = _compute_nmr(revisit_agg, baseline_agg, short_agg)

        video_result = {
            "num_revisit_pairs": len(revisit_video_pairs),
            "num_baseline_pairs": len(baseline_pairs),
            "num_short_pairs": len(short_pairs),
            "revisit": revisit_agg,
            "baseline": baseline_agg,
            "short": short_agg,
            "nmr": nmr,
        }
        all_per_video[video_name] = video_result

        # Print per-video summary
        print(f"    {'Metric':<20} {'Revisit':>10} {'Baseline':>10} {'Short':>10} {'NMR':>10}")
        print(f"    {'-'*60}")
        for metric_name in nmr:
            rev = revisit_agg.get(metric_name, {}).get("mean", 0)
            bas = baseline_agg.get(metric_name, {}).get("mean", 0)
            sh = short_agg.get(metric_name, {}).get("mean", 0)
            print(f"    {metric_name:<20} {rev:>10.4f} {bas:>10.4f} {sh:>10.4f} {nmr[metric_name]:>10.4f}")

        _save_results(output_dir, all_per_video)

    # Global summary
    print("\n" + "=" * 60)
    print("Global Summary (NMR)")
    print("=" * 60)

    global_nmr, global_revisit, global_baseline, global_short = _compute_global_summary(all_per_video)

    print(f"\n  Videos evaluated: {len(all_per_video)}")
    print(f"\n  {'Metric':<20} {'Revisit':>10} {'Baseline':>10} {'Short':>10} {'NMR':>10}")
    print(f"  {'-'*60}")
    for metric_name in global_nmr:
        rev = global_revisit.get(metric_name, 0)
        bas = global_baseline.get(metric_name, 0)
        sh = global_short.get(metric_name, 0)
        print(f"  {metric_name:<20} {rev:>10.4f} {bas:>10.4f} {sh:>10.4f} {global_nmr[metric_name]:>10.4f}")

    # Save final results
    final_output = {
        "global_summary": {
            "num_videos": len(all_per_video),
            "nmr": global_nmr,
            "revisit_means": global_revisit,
            "baseline_means": global_baseline,
            "short_means": global_short,
        },
        "per_video": all_per_video,
    }
    output_path = output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
