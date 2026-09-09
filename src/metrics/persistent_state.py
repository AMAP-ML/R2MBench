"""Persistent State Reasoning metric.

Higher-level inconsistencies that may not surface in pixel, feature, or
geometric scores. A vision-language model assigns the same pairwise
persistent-state consistency score to revisit, baseline, and short-range
pairs without being told the pair type.

The VLM caller itself is created by the eval script (via vlm_eval) and passed
into evaluate_frame_pairs_gemini, so this module has no API-client coupling.
"""

import argparse
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


EVAL_SYSTEM_PROMPT = """You are an expert evaluator of Video World Models.

Task: Assign a pairwise persistent-state consistency score to two images sampled from the same generated video.

* Image A is earlier and Image B is later.
* Do NOT assume that the images depict the same place.
* This is not raw pixel similarity: viewpoint, occlusion, and mild rendering changes may differ without implying a state inconsistency.
* Score whether the images provide evidence of the same local scene and, when they do, whether the shared scene content remains persistent.

Procedure:

1. Classify the scene relation as same, partial, different, or uncertain.
2. If shared scene content exists, estimate the viewpoint change and identify the co-visible region.
3. Judge identity, geometry, and persistence only where content can be compared, discounting differences explained by viewpoint or occlusion.
4. If there is no recognizable shared scene content, assign low consistency; absence of comparable content is not evidence of perfect consistency.

Viewpoint exemptions for same or partially overlapping scenes — do NOT penalize:
* Any difference plausibly caused by camera motion, perspective, scale, or occlusion.
* Objects entering or leaving the frame due to the viewpoint shift.
* Minor lighting, exposure, or rendering-noise differences.
* When uncertain whether a difference is from viewpoint or a true change, assume viewpoint.

Inconsistency types (report only confident findings):
* scene: images depict different local scenes or share no recognizable place cues.
* identity: same object changed color, shape, material, texture, or category.
* geometry: impossible spatial layout or structural contradictions.
* persistence: object disappeared or appeared well inside the shared region, or count of repeated elements changed without occlusion explanation.

Scoring anchors:
* 5.0 = strong evidence of the same local scene; shared state is consistent.
* 4.0 = same local scene with only minor confident inconsistencies.
* 3.0 = partial or uncertain scene overlap, or moderate inconsistencies.
* 2.0 = weak shared-scene evidence or major state inconsistencies.
* 1.0 = clearly different local scenes or fundamentally incompatible states.
Intermediate values are allowed. Do not give a high score solely because the images are individually plausible or because no content is comparable.

Output constraints — keep SHORT:
* scene_relation is one of: same, partial, different, uncertain
* viewpoint_difference ≤ 15 words
* at most 3 inconsistencies, each description ≤ 12 words
* reason ≤ 15 words
* Each inconsistency has ONLY keys: type, severity, description.

Output ONLY this JSON (no markdown):
{"scene_relation":"same|partial|different|uncertain","viewpoint_difference":"...","inconsistencies":[{"type":"scene|identity|geometry|persistence","severity":"minor|moderate|major|catastrophic","description":"..."}],"score":0.0,"reason":"..."}
"""

EVAL_USER_PROMPT = """Evaluate the pairwise persistent-state consistency of Image A and Image B. Do not assume their pair type. Output strict JSON only."""


def parse_score_response(response_text: str) -> Dict:
    """Parse Gemini JSON response and extract score (1.0–5.0)."""
    text = response_text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    response_text = text

    brace_depth = 0
    json_start = -1
    for i, char in enumerate(response_text):
        if char == '{':
            if brace_depth == 0:
                json_start = i
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0 and json_start >= 0:
                json_str = response_text[json_start:i + 1]
                try:
                    result = json.loads(json_str)
                    if "score" in result:
                        result["score"] = float(result["score"])
                        result["score"] = max(1.0, min(5.0, result["score"]))
                        return result
                except (json.JSONDecodeError, ValueError):
                    pass
                json_start = -1

    score_match = re.search(r'"score"\s*:\s*([0-9]+\.?[0-9]*)', response_text)
    if score_match:
        score = float(score_match.group(1))
        return {
            "score": max(1.0, min(5.0, score)),
            "reason": response_text[:300],
            "parse_fallback": True,
        }

    return {
        "score": -1,
        "reason": f"Failed to parse (likely truncated): {response_text[:300]}",
        "parse_error": True,
    }


def extract_frame_as_pil(video_reader: Any, frame_idx: int) -> Image.Image:
    """Extract a single frame from a VideoReader as a PIL Image."""
    frame = video_reader[frame_idx].asnumpy()
    return Image.fromarray(frame).convert("RGB")


def build_pair_images(
    video_reader: Any,
    frame_a: int,
    frame_b: int,
) -> List[Image.Image]:
    """Return [left_image, right_image] for a frame pair (no concatenation)."""
    img_a = extract_frame_as_pil(video_reader, frame_a)
    img_b = extract_frame_as_pil(video_reader, frame_b)
    return [img_a, img_b]


def evaluate_frame_pairs_gemini(
    caller,
    video_reader: Any,
    frame_pairs: List[Tuple[int, int]],
    batch_size: int = 4,
    max_retries: int = 3,
    label: str = "",
) -> List[Dict]:
    """Score a list of (frame_a, frame_b) pairs using Gemini.

    Returns a list of result dicts with at least a 'score' key.
    Failed/unparseable pairs get score=-1.
    """
    all_results: List[Dict] = [
        {"score": -1, "reason": "not evaluated"} for _ in frame_pairs
    ]
    pending_indices = list(range(len(frame_pairs)))

    for attempt in range(max_retries + 1):
        if not pending_indices:
            break
        if attempt > 0:
            print(f"    [{label}] Retry {attempt}: {len(pending_indices)} pairs pending")
            time.sleep(2)

        for batch_start in range(0, len(pending_indices), batch_size):
            batch_idx = pending_indices[batch_start:batch_start + batch_size]
            messages_list = []
            images_list = []

            for idx in batch_idx:
                fa, fb = frame_pairs[idx]
                images = build_pair_images(video_reader, fa, fb)
                messages_list.append([
                    {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": EVAL_USER_PROMPT},
                ])
                images_list.append(images)

            responses = caller.batch_call(messages_list, images_list)

            for idx, response in zip(batch_idx, responses):
                result = parse_score_response(response)
                result["raw_response"] = response
                all_results[idx] = result

        next_pending_indices = []
        for result_index in pending_indices:
            result = all_results[result_index]
            if result["score"] < 0:
                next_pending_indices.append(result_index)
        pending_indices = next_pending_indices

    valid_scores = [result["score"] for result in all_results if result["score"] > 0]
    print(f"    [{label}] {len(valid_scores)}/{len(frame_pairs)} valid scores"
          + (f", mean={sum(valid_scores)/len(valid_scores):.3f}" if valid_scores else ""))

    return all_results


def compute_gemini_nmr(
    revisit_scores: List[float],
    baseline_scores: List[float],
    short_scores: List[float],
    eps: float = 1e-8,
) -> Dict:
    """Compute NMR from Gemini scores (higher = better similarity).

    NMR = (mean_revisit - mean_baseline) / (mean_short - mean_baseline)
    """
    if not revisit_scores or not baseline_scores or not short_scores:
        return {"nmr": None, "reason": "insufficient_data"}

    mean_rev = float(np.mean(revisit_scores))
    mean_base = float(np.mean(baseline_scores))
    mean_short = float(np.mean(short_scores))

    dynamic_range = mean_short - mean_base
    if abs(dynamic_range) < eps:
        return {
            "nmr": 0.0,
            "mean_revisit": round(mean_rev, 4),
            "mean_baseline": round(mean_base, 4),
            "mean_short": round(mean_short, 4),
            "dynamic_range": round(dynamic_range, 6),
            "reason": "insufficient_dynamic_range",
        }

    nmr = (mean_rev - mean_base) / dynamic_range
    return {
        "nmr": round(float(nmr), 6),
        "mean_revisit": round(mean_rev, 4),
        "mean_baseline": round(mean_base, 4),
        "mean_short": round(mean_short, 4),
        "dynamic_range": round(dynamic_range, 6),
    }

def _load_persistent_state_cli_dependencies():
    """Load CLI-only dependencies and keep metric imports lightweight."""
    decord_module = importlib.import_module("decord")
    decord_cpu = decord_module.cpu
    video_reader_class = decord_module.VideoReader

    if __package__ in (None, ""):
        full_eval_root = Path(__file__).resolve().parents[1]
        if str(full_eval_root) not in sys.path:
            sys.path.insert(0, str(full_eval_root))

    from eval_revisit import (
        extract_revisit_pairs,
        flatten_revisit_groups,
        load_poses_from_json,
        _extract_frame_poses,
        map_pose_to_video_frame,
    )
    from eval_revisit_nmr import (
        _sample_gap_matched_temporal_baselines,
        _sample_short_temporal_pairs,
    )
    from presampled_pairs import load_pairs_json, select_for_video
    from vlm_eval import create_mllm_caller

    return {
        "decord_cpu": decord_cpu,
        "VideoReader": video_reader_class,
        "extract_revisit_pairs": extract_revisit_pairs,
        "flatten_revisit_groups": flatten_revisit_groups,
        "load_poses_from_json": load_poses_from_json,
        "_extract_frame_poses": _extract_frame_poses,
        "map_pose_to_video_frame": map_pose_to_video_frame,
        "_sample_gap_matched_temporal_baselines": _sample_gap_matched_temporal_baselines,
        "_sample_short_temporal_pairs": _sample_short_temporal_pairs,
        "load_pairs_json": load_pairs_json,
        "select_for_video": select_for_video,
        "create_mllm_caller": create_mllm_caller,
    }


def run_persistent_state_cli() -> None:
    deps = _load_persistent_state_cli_dependencies()
    decord_cpu = deps["decord_cpu"]
    video_reader_class = deps["VideoReader"]
    extract_revisit_pairs = deps["extract_revisit_pairs"]
    flatten_revisit_groups = deps["flatten_revisit_groups"]
    load_poses_from_json = deps["load_poses_from_json"]
    _extract_frame_poses = deps["_extract_frame_poses"]
    map_pose_to_video_frame = deps["map_pose_to_video_frame"]
    _sample_gap_matched_temporal_baselines = deps["_sample_gap_matched_temporal_baselines"]
    _sample_short_temporal_pairs = deps["_sample_short_temporal_pairs"]
    load_pairs_json = deps["load_pairs_json"]
    select_for_video = deps["select_for_video"]
    create_mllm_caller = deps["create_mllm_caller"]

    parser = argparse.ArgumentParser(
        description="Gemini-NMR: Normalized Memory Retention via Gemini VLM"
    )
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing video files (*.mp4 or */gen.mp4 or */output.mp4)")
    parser.add_argument("--pose_json", type=str, required=True,
                        help="Global trajectory pose JSON")
    parser.add_argument("--output_json", type=str, default="./results_gemini_nmr.json",
                        help="Output JSON file")
    parser.add_argument("--engine", type=str, default="gemini-3.1-pro-preview")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Parallel API workers")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_pairs_per_type", type=int, default=15,
                        help="Max pairs per category (revisit/baseline/short) per video")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--angle_tolerance", type=float, default=None,
                        help="Yaw tolerance in degrees (default: adaptive)")
    parser.add_argument("--translation_tolerance", type=float, default=None,
                        help="Translation tolerance in meters (default: adaptive)")
    parser.add_argument("--min_gap_ratio", type=float, default=0.2,
                        help="Minimum revisit gap as fraction of video length")
    parser.add_argument("--short_max_gap", type=int, default=None,
                        help="Max frame gap for short pairs (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-evaluated videos in output_json")
    parser.add_argument("--pairs_json", type=str, default=None,
                        help="Pre-sampled shared pairs JSON (revisit/baseline/short "
                             "in video-frame space). When set, these pairs are used "
                             "instead of per-video internal sampling so all eval "
                             "types score identical pairs.")
    args = parser.parse_args()


    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load existing results
    all_results: Dict[str, Dict] = {}
    if args.resume and output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            all_results = existing.get("videos", {})
            print(f"Resumed from {output_path}: {len(all_results)} videos already done")
        except Exception as exc:
            print(f"Warning: could not load existing results ({exc}), starting fresh")

    # Load poses
    print(f"Loading pose JSON: {args.pose_json}")
    pose_data, sorted_keys, _ = load_poses_from_json(args.pose_json)
    frame_poses = _extract_frame_poses(pose_data, sorted_keys)
    n_pose = len(sorted_keys)

    revisit_groups, _ = extract_revisit_pairs(
        args.pose_json,
        angle_tolerance=args.angle_tolerance,
        translation_tolerance=args.translation_tolerance,
    )
    all_revisit_pairs = flatten_revisit_groups(revisit_groups)
    print(f"Pose frames: {n_pose}, Revisit pairs detected: {len(all_revisit_pairs)}")
    if not all_revisit_pairs and args.pairs_json is None:
        print("ERROR: No revisit pairs found in trajectory!")
        sys.exit(1)

    # Load pre-sampled shared pairs (unified pipeline). When set, all eval types
    # score identical revisit/baseline/short pairs.
    presampled_pairs = None
    if args.pairs_json:
        presampled_pairs = load_pairs_json(args.pairs_json)
        print(f"Using pre-sampled pairs from {args.pairs_json}: "
              f"revisit={len(presampled_pairs['revisit'])}, "
              f"baseline={len(presampled_pairs['baseline'])}, "
              f"short={len(presampled_pairs['short'])}")

    # Discover videos
    video_dir = Path(args.video_dir)
    all_videos = sorted(video_dir.glob("*.mp4"))
    if not all_videos:
        all_videos = sorted(video_dir.glob("*/gen.mp4"))
    if not all_videos:
        all_videos = sorted(video_dir.glob("*/output.mp4"))
    if not all_videos:
        print(f"ERROR: No .mp4 files found in {video_dir}")
        sys.exit(1)

    if args.max_videos:
        all_videos = all_videos[:args.max_videos]
    print(f"Found {len(all_videos)} videos")

    # Create Gemini caller
    print(f"Using engine: {args.engine}, num_workers={args.num_workers}")
    caller = create_mllm_caller(
        engine=args.engine,
        temperature=0.0,
        max_tokens=4096,
        num_clients=args.num_workers,
    )

    max_pairs = args.max_pairs_per_type
    global_revisit_scores: List[float] = []
    global_baseline_scores: List[float] = []
    global_short_scores: List[float] = []

    for vid_idx, video_path in enumerate(all_videos):
        # Determine video name
        if video_path.name in ("gen.mp4", "output.mp4"):
            video_name = video_path.parent.name
        else:
            video_name = video_path.stem

        if video_name in all_results:
            print(f"\n[{vid_idx+1}/{len(all_videos)}] {video_name} - SKIPPED (resumed)")
            # Accumulate global scores from resumed results
            vdata = all_results[video_name]
            for score in vdata.get("revisit_scores", []):
                if score > 0:
                    global_revisit_scores.append(score)
            for score in vdata.get("baseline_scores", []):
                if score > 0:
                    global_baseline_scores.append(score)
            for score in vdata.get("short_scores", []):
                if score > 0:
                    global_short_scores.append(score)
            continue

        print(f"\n{'='*55}")
        print(f"[{vid_idx+1}/{len(all_videos)}] {video_name}")

        # Open video
        try:
            vr = video_reader_class(str(video_path), ctx=decord_cpu(0))
        except Exception as exc:
            print(f"  ERROR opening video: {exc}")
            continue

        n_video = len(vr)
        effective_frames = min(n_video, n_pose)

        if presampled_pairs is not None:
            # Unified pipeline: use shared pre-sampled pairs (video-frame space).
            sampled_revisit, sampled_baseline, sampled_short = select_for_video(
                presampled_pairs, n_video, max_per_type=max_pairs
            )
            if not sampled_revisit:
                print(f"  No valid pre-sampled revisit pairs for this video")
                del vr
                continue
            baseline_stats = {"source": "presampled"}
            short_stats = {"source": "presampled", "min_gap": 2, "max_gap": args.short_max_gap}
            print(f"  [pre-sampled] revisit={len(sampled_revisit)}, "
                  f"baseline={len(sampled_baseline)}, short={len(sampled_short)}")
        else:
            min_gap = int(effective_frames * args.min_gap_ratio)

            # Map revisit pairs to video frame space and apply gap filter
            revisit_video_pairs: List[Tuple[int, int]] = []
            for fa, fb, _, _ in all_revisit_pairs:
                vid_a = map_pose_to_video_frame(fa, n_pose, n_video)
                vid_b = map_pose_to_video_frame(fb, n_pose, n_video)
                if vid_a >= effective_frames or vid_b >= effective_frames:
                    continue
                if abs(vid_b - vid_a) >= min_gap:
                    revisit_video_pairs.append((vid_a, vid_b))

            if not revisit_video_pairs:
                print(f"  No valid revisit pairs after gap filter (min_gap={min_gap})")
                del vr
                continue

            # Sample revisit pairs (up to max_pairs)
            if len(revisit_video_pairs) > max_pairs:
                sampled_revisit = sorted(
                    revisit_video_pairs,
                    key=lambda pair: hashlib.sha256(
                        f"{args.seed}:{pair[0]}:{pair[1]}".encode("utf-8")
                    ).hexdigest(),
                )[:max_pairs]
            else:
                sampled_revisit = list(revisit_video_pairs)
            print(f"  Revisit pairs: {len(sampled_revisit)}/{len(revisit_video_pairs)} sampled")

            # Sample baseline (gap-matched temporal negatives)
            exclusion_radius = max(10, effective_frames // 30)
            baseline_all, baseline_stats = _sample_gap_matched_temporal_baselines(
                sampled_revisit, frame_poses, effective_frames,
                num_temporal_bins=10,
                exclusion_radius=exclusion_radius,
                seed=args.seed,
            )
            sampled_baseline = baseline_all[:max_pairs]
            print(f"  Baseline pairs: {len(sampled_baseline)} "
                  f"(match_rate={baseline_stats.get('matched_rate', 0):.2%})")

            # Sample short pairs
            sampled_short, short_stats = _sample_short_temporal_pairs(
                frame_poses, effective_frames,
                count=max_pairs,
                min_gap=2,
                max_gap=args.short_max_gap,
                seed=args.seed,
            )
            print(f"  Short pairs: {len(sampled_short)} "
                  f"(gap=[{short_stats['min_gap']},{short_stats['max_gap']}])")

        # Evaluate all three types with Gemini
        start_time = time.time()

        revisit_raw = evaluate_frame_pairs_gemini(
            caller, vr, sampled_revisit,
            batch_size=args.batch_size, label="revisit"
        )
        baseline_raw = evaluate_frame_pairs_gemini(
            caller, vr, sampled_baseline,
            batch_size=args.batch_size, label="baseline"
        )
        short_raw = evaluate_frame_pairs_gemini(
            caller, vr, sampled_short,
            batch_size=args.batch_size, label="short"
        )

        elapsed = time.time() - start_time
        del vr

        # Extract valid scores
        revisit_scores = [r["score"] for r in revisit_raw if r["score"] > 0]
        baseline_scores = [r["score"] for r in baseline_raw if r["score"] > 0]
        short_scores = [r["score"] for r in short_raw if r["score"] > 0]

        global_revisit_scores.extend(revisit_scores)
        global_baseline_scores.extend(baseline_scores)
        global_short_scores.extend(short_scores)

        # Compute per-video NMR
        nmr_result = compute_gemini_nmr(revisit_scores, baseline_scores, short_scores)
        nmr_value = nmr_result.get("nmr")

        print(f"  Scores: revisit={nmr_result.get('mean_revisit', 'N/A')}, "
              f"baseline={nmr_result.get('mean_baseline', 'N/A')}, "
              f"short={nmr_result.get('mean_short', 'N/A')}")
        print(f"  NMR = {nmr_value}  (elapsed={elapsed:.1f}s)")

        # Save per-video result
        all_results[video_name] = {
            "revisit_pairs": sampled_revisit,
            "baseline_pairs": sampled_baseline,
            "short_pairs": sampled_short,
            "revisit_scores": [r["score"] for r in revisit_raw],
            "baseline_scores": [r["score"] for r in baseline_raw],
            "short_scores": [r["score"] for r in short_raw],
            "revisit_details": revisit_raw,
            "baseline_details": baseline_raw,
            "short_details": short_raw,
            "nmr": nmr_result,
            "evaluation_time_sec": round(elapsed, 2),
        }

        # Incremental save (resume-safe)
        _save_incremental(output_path, args, all_results,
                          global_revisit_scores, global_baseline_scores, global_short_scores)

    # Global NMR summary
    print(f"\n{'='*60}")
    print("GLOBAL SUMMARY")
    print(f"{'='*60}")

    global_nmr = compute_gemini_nmr(
        global_revisit_scores, global_baseline_scores, global_short_scores
    )
    print(f"  Videos evaluated : {len(all_results)}")
    print(f"  mean_revisit     : {global_nmr.get('mean_revisit', 'N/A')}")
    print(f"  mean_baseline    : {global_nmr.get('mean_baseline', 'N/A')}")
    print(f"  mean_short       : {global_nmr.get('mean_short', 'N/A')}")
    print(f"  NMR_gemini       : {global_nmr.get('nmr', 'N/A')}")

    # Final save
    _save_incremental(output_path, args, all_results,
                      global_revisit_scores, global_baseline_scores, global_short_scores)
    print(f"\nResults saved to: {output_path}")


def _save_incremental(
    output_path: Path,
    args,
    all_results: Dict,
    global_revisit_scores: List[float],
    global_baseline_scores: List[float],
    global_short_scores: List[float],
):
    """Save current state to JSON (incremental/resume-safe)."""
    global_nmr = compute_gemini_nmr(
        global_revisit_scores, global_baseline_scores, global_short_scores
    )
    payload = {
        "engine": args.engine,
        "max_pairs_per_type": args.max_pairs_per_type,
        "global_nmr": global_nmr,
        "num_videos": len(all_results),
        "videos": all_results,
    }
    temp_path = output_path.with_suffix(".json.tmp")
    with open(temp_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, output_path)


if __name__ == "__main__":
    run_persistent_state_cli()
