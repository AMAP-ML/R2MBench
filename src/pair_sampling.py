"""Standalone pair sampling utilities for pre-sampling revisit/baseline/short pairs.

This module is a self-contained copy of the sampling functions originally defined
in eval_revisit_nmr.py. It exists so that sample_pairs.py and the three eval
scripts can share the same sampling logic without importing from eval_revisit_nmr
(which may have its own modifications or dependencies).

Functions:
    _angle_diff_deg
    _translation_distance
    _make_temporal_bins
    _get_bin_index
    _candidate_frames_from_expanded_bins
    sample_gap_matched_temporal_baselines
    sample_short_temporal_pairs
"""

import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two yaw angles in degrees."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def _translation_distance(p1: dict, p2: dict) -> float:
    """Euclidean translation distance between two frame poses."""
    t1 = np.array([p1["tx"], p1["ty"], p1["tz"]], dtype=np.float64)
    t2 = np.array([p2["tx"], p2["ty"], p2["tz"]], dtype=np.float64)
    return float(np.linalg.norm(t1 - t2))


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


def sample_gap_matched_temporal_baselines(
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


def sample_short_temporal_pairs(
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
