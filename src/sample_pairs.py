#!/usr/bin/env python3
"""Pre-sample revisit/baseline/short pairs for all tasks.

Outputs one JSON file per unique pose_json, containing pre-sampled frame
indices that can be shared across NMR, Object Consistency, and Gemini evals.

Usage:
    python sample_pairs.py --tasks_conf full_eval/tasks.conf --output_dir full_eval/pairs
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault(
    "TORCH_HOME",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "torch_hub_cache"),
)

import numpy as np
from decord import VideoReader, cpu as decord_cpu

# Import the vendored eval_revisit / pair_sampling that live alongside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_revisit import (
    extract_revisit_pairs,
    flatten_revisit_groups,
    load_poses_from_json,
    map_pose_to_video_frame,
    _extract_frame_poses,
)
from pair_sampling import (
    sample_gap_matched_temporal_baselines,
    sample_short_temporal_pairs,
)

REVISIT_SAMPLING_VERSION = "adaptive-p95-v1"


def parse_tasks_conf(conf_path: str):
    """Parse tasks.conf, return list of (name, video_dir, pose_json)."""
    tasks = []
    with open(conf_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            name, video_dir, pose_json = [p.strip() for p in parts]
            tasks.append((name, video_dir, pose_json))
    return tasks


def find_representative_video(video_dir: str) -> Path | None:
    """Find first available mp4 in video_dir."""
    vdir = Path(video_dir)
    if not vdir.exists():
        return None
    mp4s = sorted(vdir.glob("*.mp4"))
    if mp4s:
        return mp4s[0]
    # Try subdirectory pattern
    for subdir in sorted(vdir.iterdir()):
        if subdir.is_dir():
            gen = subdir / "gen.mp4"
            if gen.exists():
                return gen
            out = subdir / "output.mp4"
            if out.exists():
                return out
    return None


def pose_json_key(pose_json: str) -> str:
    """Generate a short hash key for a pose_json path."""
    h = hashlib.md5(pose_json.encode()).hexdigest()[:8]
    name = Path(pose_json).stem
    return f"{name}_{h}"


def sample_pairs_for_pose(
    pose_json: str,
    representative_video: Path,
    max_eval_pairs: int = 100,
    seed: int = 42,
    min_gap_ratio: float = 0.2,
    angle_tolerance: Optional[float] = None,
    translation_tolerance: Optional[float] = None,
    short_multiplier: float = 1.0,
):
    """Sample revisit/baseline/short pairs for a given pose_json.

    Returns dict with keys: revisit, baseline, short (each a list of [va, vb]).
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # Extract revisit pairs from pose
    revisit_groups, n_pose, threshold_info = extract_revisit_pairs(
        pose_json,
        angle_tolerance=angle_tolerance,
        translation_tolerance=translation_tolerance,
        return_thresholds=True,
    )
    flat_pairs = flatten_revisit_groups(revisit_groups)

    # Load frame poses for baseline sampling
    pose_data, sorted_keys, _ = load_poses_from_json(pose_json)
    frame_poses = _extract_frame_poses(pose_data, sorted_keys)

    # Get n_video from representative video
    vr = VideoReader(str(representative_video), ctx=decord_cpu(0))
    n_video = len(vr)
    del vr

    effective_frames = min(n_video, n_pose) if n_pose else n_video
    min_gap = int(effective_frames * min_gap_ratio)

    print(f"  n_pose={n_pose}, n_video={n_video}, effective={effective_frames}, min_gap={min_gap}")
    print(f"  Raw revisit pairs: {len(flat_pairs)}")

    # Filter revisit pairs by gap
    filtered_revisit = []
    for fa, fb, dpos, drot in flat_pairs:
        vid_a = map_pose_to_video_frame(fa, n_pose, n_video)
        vid_b = map_pose_to_video_frame(fb, n_pose, n_video)
        if vid_a >= effective_frames or vid_b >= effective_frames:
            continue
        if abs(vid_b - vid_a) >= min_gap:
            filtered_revisit.append((fa, fb, dpos, drot))

    print(f"  After gap filter: {len(filtered_revisit)}")

    if not filtered_revisit:
        return {"revisit": [], "baseline": [], "short": [], "meta": {"error": "no valid revisit pairs"}}

    # Sample revisit pairs
    sampled_revisit = filtered_revisit
    if max_eval_pairs and len(filtered_revisit) > max_eval_pairs:
        sampled_revisit = rng.sample(filtered_revisit, max_eval_pairs)

    # Convert to video-frame space
    revisit_video_pairs = []
    for fa, fb, _, _ in sampled_revisit:
        vid_a = map_pose_to_video_frame(fa, n_pose, n_video)
        vid_b = map_pose_to_video_frame(fb, n_pose, n_video)
        revisit_video_pairs.append([vid_a, vid_b])

    print(f"  Sampled revisit: {len(revisit_video_pairs)}")

    # Sample baseline pairs (use conservative exclusion_radius)
    exclusion_radius = max(10, effective_frames // 20)
    num_temporal_bins = 10
    gap_tolerance_ratio = 0.3

    baseline_pairs_raw, baseline_stats = sample_gap_matched_temporal_baselines(
        [(va, vb) for va, vb in revisit_video_pairs],
        frame_poses,
        effective_frames,
        num_temporal_bins=num_temporal_bins,
        exclusion_radius=exclusion_radius,
        gap_tolerance_ratio=gap_tolerance_ratio,
        seed=seed,
    )
    baseline_video_pairs = [[va, vb] for va, vb in baseline_pairs_raw]
    print(f"  Baseline: {len(baseline_video_pairs)} (match_rate={baseline_stats.get('matched_rate', 0):.2%})")

    # Sample short pairs
    short_count = min(100, max(20, int(len(revisit_video_pairs) * short_multiplier)))
    short_pairs_raw, short_stats = sample_short_temporal_pairs(
        frame_poses,
        effective_frames,
        count=short_count,
        min_gap=2,
        seed=seed,
    )
    short_video_pairs = [[va, vb] for va, vb in short_pairs_raw]
    print(f"  Short: {len(short_video_pairs)}/{short_stats['target_count']}")

    return {
        "revisit": revisit_video_pairs,
        "baseline": baseline_video_pairs,
        "short": short_video_pairs,
        "meta": {
            "pose_json": pose_json,
            "representative_video": str(representative_video),
            "n_pose": n_pose,
            "n_video": n_video,
            "effective_frames": effective_frames,
            "seed": seed,
            "max_eval_pairs": max_eval_pairs,
            "min_gap_ratio": min_gap_ratio,
            "revisit_threshold_mode": threshold_info["mode"],
            "revisit_sampling_version": REVISIT_SAMPLING_VERSION,
            "angle_tolerance_deg": threshold_info["angle_tolerance_deg"],
            "translation_tolerance_m": threshold_info["translation_tolerance_m"],
            "exclusion_radius": exclusion_radius,
            "baseline_stats": baseline_stats,
            "short_stats": short_stats,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-sample pairs for all tasks")
    parser.add_argument("--tasks_conf", type=str, default="full_eval/tasks.conf")
    parser.add_argument("--output_dir", type=str, default="full_eval/pairs")
    parser.add_argument("--max_eval_pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = parse_tasks_conf(args.tasks_conf)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group tasks by pose_json
    pose_to_tasks: dict[str, list[tuple[str, str]]] = {}
    for name, video_dir, pose_json in tasks:
        pose_to_tasks.setdefault(pose_json, []).append((name, video_dir))

    print(f"Tasks: {len(tasks)}, Unique poses: {len(pose_to_tasks)}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    results_summary = {}

    for pose_json, task_list in pose_to_tasks.items():
        key = pose_json_key(pose_json)
        out_path = output_dir / f"{key}.json"

        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
                existing_meta = existing.get("meta", {})
                if (
                    existing_meta.get("revisit_threshold_mode") == "adaptive"
                    and existing_meta.get("revisit_sampling_version")
                    == REVISIT_SAMPLING_VERSION
                ):
                    print(f"\n[SKIP] {key} already exists with adaptive thresholds "
                          f"({len(task_list)} tasks)")
                    results_summary[key] = {
                        "revisit": len(existing.get("revisit", [])),
                        "baseline": len(existing.get("baseline", [])),
                        "short": len(existing.get("short", [])),
                        "tasks": [t[0] for t in task_list],
                    }
                    continue
                print(f"\n[STALE] {key} uses fixed or unknown revisit thresholds; resampling")
            except Exception:
                print(f"\n[STALE] {key} is unreadable; resampling")

        if not Path(pose_json).exists():
            print(f"\n[ERROR] Pose file not found: {pose_json}")
            continue

        # Find representative video from any task sharing this pose
        rep_video = None
        for name, video_dir in task_list:
            rv = find_representative_video(video_dir)
            if rv is not None:
                rep_video = rv
                break

        if rep_video is None:
            print(f"\n[ERROR] No representative video for pose {key}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Sampling pairs for: {key}")
        print(f"  Pose: {pose_json}")
        print(f"  Rep video: {rep_video}")
        print(f"  Tasks: {[t[0] for t in task_list]}")
        print(f"{'=' * 60}")

        pairs = sample_pairs_for_pose(
            pose_json,
            rep_video,
            max_eval_pairs=args.max_eval_pairs,
            seed=args.seed,
        )

        out_path.write_text(json.dumps(pairs, indent=2))
        print(f"  Saved to: {out_path}")

        results_summary[key] = {
            "revisit": len(pairs["revisit"]),
            "baseline": len(pairs["baseline"]),
            "short": len(pairs["short"]),
            "tasks": [t[0] for t in task_list],
        }

    # Write summary
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results_summary, indent=2))

    print(f"\n{'=' * 60}")
    print("Summary:")
    for key, info in results_summary.items():
        print(f"  {key}: rev={info['revisit']}, base={info['baseline']}, "
              f"short={info['short']}, tasks={len(info['tasks'])}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
