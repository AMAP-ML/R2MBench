#!/usr/bin/env python3
"""Unified one-pass GPU NMR evaluator.

Computes any subset of the four GPU metric families in a SINGLE per-pair pass
over the shared revisit / baseline / short pairs -- one frame decode serves all
requested metrics:

    appearance      PSNR, SSIM (+ LPIPS with --enable_lpips)
    scene_identity  DINOv2 + BoQ + MutualVPR global-descriptor similarity
    geometric       SuperPoint + LightGlue matches + two-view geometry
    object          GroundingDINO + SAM2 + DINOv2 (+ CLIP) object persistence

Only the models needed by the requested --families are loaded. All heavy logic
(pose/pair detection, gap-matched baseline & short sampling, pre-sampled pair
selection, resume, atomic partial saves, the visual metric functions, and the
object evaluator) is reused from eval_revisit_nmr / metrics / metrics.object_identity;
this script only adds the merged per-pair loop and a union-key aggregation so all
four families share identical pairs and are reported under one flat NMR schema.

Video sharding (--num_video_shards / --video_shard_id) is used by run_parallel_eval.sh
to spread work across GPUs (batch x shard processes).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

# Reuse the existing evaluator internals. Importing this module also wires up
# metrics/* (extractors, metric fns) and the pose/pair helpers.
import eval_revisit_nmr as ern
from metrics.object_identity import (
    GroundedSamObjectEvaluator,
    CLIPFeatureExtractor,
    DEFAULT_TEXT_PROMPT,
)

VALID_FAMILIES = {"appearance", "scene_identity", "geometric", "object"}

VISUAL_KEYS = [
    "psnr", "ssim", "dino_similarity", "boq_similarity", "mvpr_similarity",
    "lpips", "num_matches", "match_ratio", "ransac_inlier_ratio",
]
OBJECT_KEYS = [
    "object_existence_rate", "object_appearance_similarity",
    "object_semantic_similarity", "object_appearance_persistence",
    "object_semantic_persistence", "object_recall",
    "object_identity_similarity", "object_consistency",
    "num_anchor_objects", "num_detected_candidates", "num_object_matches",
]
ALL_KEYS = VISUAL_KEYS + OBJECT_KEYS

# NMR direction / exclusions (union of both scripts' conventions).
LOWER_IS_BETTER = {"lpips"}
SKIP_NMR = {
    "num_matches", "num_anchor_objects", "num_detected_candidates",
    "num_object_matches",
}
MIN_DYNAMIC_RANGE = 1e-5


# ---------------------------------------------------------------------------
# Aggregation / NMR over the union key set (mirrors eval_revisit_nmr, extended)
# ---------------------------------------------------------------------------
def _aggregate(results):
    agg = {}
    for key in ALL_KEYS:
        values = [r[key] for r in results if r.get("status") == "ok" and key in r]
        if values:
            agg[key] = {
                "mean": round(float(np.mean(values)), 6),
                "std": round(float(np.std(values)), 6),
                "median": round(float(np.median(values)), 6),
                "count": len(values),
            }
    return agg


def _compute_nmr(revisit_agg, baseline_agg, short_agg):
    nmr = {}
    for key in revisit_agg:
        if key in SKIP_NMR:
            continue
        if key not in baseline_agg or key not in short_agg:
            continue
        r = revisit_agg[key]["mean"]
        b = baseline_agg[key]["mean"]
        s = short_agg[key]["mean"]
        if key in LOWER_IS_BETTER:
            dyn = b - s
            num = b - r
        else:
            dyn = s - b
            num = r - b
        if dyn <= MIN_DYNAMIC_RANGE:
            nmr[key] = 0.0
        else:
            nmr[key] = round(float(num / dyn), 6)
    return nmr


def _global_summary(all_per_video):
    rev_t = {k: [] for k in ALL_KEYS}
    base_t = {k: [] for k in ALL_KEYS}
    short_t = {k: [] for k in ALL_KEYS}
    for vr in all_per_video.values():
        if not isinstance(vr, dict):
            continue
        for key in ALL_KEYS:
            if isinstance(vr.get("revisit", {}).get(key), dict):
                rev_t[key].append(vr["revisit"][key]["mean"])
            if isinstance(vr.get("baseline", {}).get(key), dict):
                base_t[key].append(vr["baseline"][key]["mean"])
            if isinstance(vr.get("short", {}).get(key), dict):
                short_t[key].append(vr["short"][key]["mean"])
    g_rev, g_base, g_short, g_nmr = {}, {}, {}, {}
    for key in ALL_KEYS:
        if rev_t[key]:
            g_rev[key] = round(float(np.mean(rev_t[key])), 6)
        if base_t[key]:
            g_base[key] = round(float(np.mean(base_t[key])), 6)
        if short_t[key]:
            g_short[key] = round(float(np.mean(short_t[key])), 6)
        if key in SKIP_NMR:
            continue
        if rev_t[key] and base_t[key] and short_t[key]:
            r = float(np.mean(rev_t[key]))
            b = float(np.mean(base_t[key]))
            s = float(np.mean(short_t[key]))
            if key in LOWER_IS_BETTER:
                dyn, num = (b - s), (b - r)
            else:
                dyn, num = (s - b), (r - b)
            if abs(dyn) > 1e-8:
                g_nmr[key] = round(num / dyn, 6)
    return g_nmr, g_rev, g_base, g_short


# ---------------------------------------------------------------------------
# Merged per-pair loop: decode once, run all requested families
# ---------------------------------------------------------------------------
def _evaluate_pairs(vr, pairs, *, dino_ext, boq_ext, mvpr_ext, keypoint_matcher,
                    skip_pixel_metrics, lpips_clip, object_evaluator, label):
    results = []
    for i, (a, b) in enumerate(pairs):
        img_a = vr[a].asnumpy()
        img_b = vr[b].asnumpy()
        entry = {"status": "ok", "video_frames": [int(a), int(b)]}

        if not skip_pixel_metrics:
            entry.update(ern.compute_psnr_ssim(img_a, img_b))
        if dino_ext is not None:
            entry["dino_similarity"] = round(dino_ext.similarity(img_a, img_b), 6)
        if boq_ext is not None:
            entry["boq_similarity"] = round(boq_ext.similarity(img_a, img_b), 6)
        if mvpr_ext is not None:
            entry["mvpr_similarity"] = round(mvpr_ext.similarity(img_a, img_b), 6)
        if lpips_clip is not None:
            entry["lpips"] = round(lpips_clip.compute_lpips(img_a, img_b), 6)
        if keypoint_matcher is not None:
            kp = keypoint_matcher.match(img_a, img_b)
            entry["num_keypoints_a"] = kp["num_keypoints_a"]
            entry["num_keypoints_b"] = kp["num_keypoints_b"]
            entry["num_matches"] = kp["num_matches"]
            entry["match_ratio"] = kp["match_ratio"]
            entry.update(ern.compute_two_view_geometry(
                kp.get("matched_kp_a"), kp.get("matched_kp_b")))
        if object_evaluator is not None:
            score = object_evaluator.score_pair(img_a, img_b, int(a), int(b))
            score.pop("anchor_objects", None)   # drop bulky per-pair diagnostics
            score.pop("object_matches", None)
            entry.update(score)

        results.append(entry)
        if (i + 1) % 10 == 0:
            print(f"      {label}: {i + 1}/{len(pairs)} done", flush=True)
    return results


def _parse_families(raw):
    toks = [t.strip().lower() for t in raw.split(",") if t.strip()]
    if not toks or "all" in toks:
        return set(VALID_FAMILIES)
    fams = set()
    for t in toks:
        if t not in VALID_FAMILIES:
            raise SystemExit(
                f"--families: unknown family '{t}'. "
                f"Valid: appearance, scene_identity, geometric, object, all.")
        fams.add(t)
    return fams


def main():
    p = argparse.ArgumentParser(description="Unified one-pass GPU NMR evaluation")
    # Core (shared with eval_revisit_nmr)
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--pose_json", type=str, default=None)
    p.add_argument("--per_video_pose", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./eval_gpu_results")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--angle_tolerance", type=float, default=None,
                   help="Max yaw difference in degrees (default: adaptive)")
    p.add_argument("--translation_tolerance", type=float, default=None,
                   help="Max translation difference in meters (default: adaptive)")
    p.add_argument("--max_videos", type=int, default=None)
    p.add_argument("--num_video_shards", type=int, default=1)
    p.add_argument("--video_shard_id", type=int, default=0)
    p.add_argument("--enable_lpips", action="store_true")
    p.add_argument("--clip_checkpoint", type=str, default=None)
    p.add_argument("--max_eval_pairs", type=int, default=100)
    p.add_argument("--short_multiplier", type=float, default=1.0)
    p.add_argument("--short_max_gap", type=int, default=3)
    p.add_argument("--min_gap_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pairs_json", type=str, default=None)
    p.add_argument("--families", type=str, default="all")
    # Object family (required only when 'object' is selected)
    p.add_argument("--groundingdino_config", type=str, default=None)
    p.add_argument("--groundingdino_checkpoint", type=str, default=None)
    p.add_argument("--sam2_config", type=str, default=None)
    p.add_argument("--sam_checkpoint", type=str, default=None)
    p.add_argument("--text_prompt", type=str, default=DEFAULT_TEXT_PROMPT)
    p.add_argument("--box_threshold", type=float, default=0.25)
    p.add_argument("--text_threshold", type=float, default=0.20)
    p.add_argument("--nms_iou", type=float, default=0.60)
    p.add_argument("--max_objects", type=int, default=10)
    p.add_argument("--max_candidates_per_concept", type=int, default=5)
    p.add_argument("--min_area_ratio", type=float, default=0.002)
    p.add_argument("--match_threshold", type=float, default=0.50)
    p.add_argument("--appearance_weight", type=float, default=0.5)
    p.add_argument("--semantic_weight", type=float, default=0.5)
    args = p.parse_args()

    if not args.pose_json and not args.per_video_pose:
        p.error("Either --pose_json or --per_video_pose must be specified")
    if args.num_video_shards < 1:
        p.error("--num_video_shards must be >= 1")
    if not (0 <= args.video_shard_id < args.num_video_shards):
        p.error("--video_shard_id must be in [0, num_video_shards)")

    families = _parse_families(args.families)
    want_appearance = "appearance" in families
    want_scene = "scene_identity" in families
    want_geometric = "geometric" in families
    want_object = "object" in families
    skip_pixel_metrics = not want_appearance
    if not want_appearance:
        args.enable_lpips = False
    if want_object:
        missing = [f for f in ("groundingdino_config", "groundingdino_checkpoint",
                               "sam2_config", "sam_checkpoint")
                   if not getattr(args, f)]
        if missing:
            p.error(f"object family requires: {', '.join('--' + m for m in missing)}")

    print(f"  Selected metric families: {sorted(families)}", flush=True)

    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    resumed_results = {}
    if args.resume:
        for fname in ("results.json", "results_partial.json"):
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

    presampled_pairs = None
    if args.pairs_json:
        presampled_pairs = ern.load_pairs_json(args.pairs_json)
        print(f"  Using pre-sampled pairs from {args.pairs_json}: "
              f"revisit={len(presampled_pairs['revisit'])}, "
              f"baseline={len(presampled_pairs['baseline'])}, "
              f"short={len(presampled_pairs['short'])}", flush=True)

    # ---- Load only the requested families' models -----------------------
    print("=" * 60 + "\nLoading models...\n" + "=" * 60, flush=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    dino_ext = boq_ext = mvpr_ext = None
    if want_scene:
        dino_ext = ern.DINOv2FeatureExtractor(device=device)
        boq_ext = ern.BoQFeatureExtractor(device=device)
        mvpr_ext = ern.MutualVPRFeatureExtractor(device=device)
    keypoint_matcher = ern.KeypointMatcher(device=device) if want_geometric else None
    lpips_clip = None
    if args.enable_lpips:
        lpips_clip = ern.LpipsClipComputer(clip_checkpoint_path=args.clip_checkpoint, device=device)

    object_evaluator = None
    if want_object:
        # Share the DINOv2 instance with scene_identity when both are selected.
        obj_dino = dino_ext if dino_ext is not None else ern.DINOv2FeatureExtractor(device=device)
        obj_clip = CLIPFeatureExtractor(args.clip_checkpoint, device=device) if args.clip_checkpoint else None
        object_evaluator = GroundedSamObjectEvaluator(
            groundingdino_config=args.groundingdino_config,
            groundingdino_checkpoint=args.groundingdino_checkpoint,
            sam_checkpoint=args.sam_checkpoint,
            sam2_config=args.sam2_config,
            dino_extractor=obj_dino,
            clip_extractor=obj_clip,
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

    # ---- Discover + shard videos ----------------------------------------
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
    num_discovered = len(all_videos)
    all_videos = all_videos[args.video_shard_id::args.num_video_shards]
    print(f"  Found {num_discovered} videos; shard "
          f"{args.video_shard_id}/{args.num_video_shards} -> {len(all_videos)} videos", flush=True)

    per_video_pose_mode = args.per_video_pose is not None

    def get_pose_and_pairs(pose_json_path):
        revisit_groups, n_pose = ern.extract_revisit_pairs(
            pose_json_path, angle_tolerance=args.angle_tolerance,
            translation_tolerance=args.translation_tolerance)
        flat = ern.flatten_revisit_groups(revisit_groups)
        pose_data, sorted_keys, _ = ern.load_poses_from_json(pose_json_path)
        frame_poses = ern._extract_frame_poses(pose_data, sorted_keys)
        return flat, n_pose, frame_poses

    global_pairs = global_n_pose = global_frame_poses = None
    if not per_video_pose_mode:
        global_pairs, global_n_pose, global_frame_poses = get_pose_and_pairs(args.pose_json)
        print(f"  Poses: {global_n_pose}, Revisit pairs: {len(global_pairs)}", flush=True)
        if not global_pairs and presampled_pairs is None:
            print("  ERROR: No revisit pairs detected!")
            sys.exit(1)

    all_per_video = dict(resumed_results)

    for vid_idx, video_path in enumerate(all_videos):
        video_name = (video_path.parent.name
                      if video_path.name in ("gen.mp4", "output.mp4")
                      else video_path.stem)
        if video_name in resumed_results:
            print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - SKIPPED (resumed)")
            continue

        if per_video_pose_mode:
            pose_path = video_path.parent / args.per_video_pose
            if not pose_path.exists():
                print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - SKIPPED (no pose)")
                continue
            revisit_pairs, n_pose, frame_poses = get_pose_and_pairs(str(pose_path))
        else:
            revisit_pairs, n_pose, frame_poses = global_pairs, global_n_pose, global_frame_poses

        if not revisit_pairs and presampled_pairs is None:
            print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name} - SKIPPED (no revisit pairs)")
            continue

        print(f"\n  [{vid_idx+1}/{len(all_videos)}] {video_name}", flush=True)
        try:
            vr = ern.VideoReader(str(video_path), ctx=ern.decord_cpu(0))
        except Exception as exc:
            print(f"    ERROR opening video: {exc}")
            continue
        n_video = len(vr)
        effective_frames = min(n_video, n_pose) if n_pose else n_video

        if presampled_pairs is not None:
            revisit_vp, baseline_pairs, short_pairs = ern.select_for_video(
                presampled_pairs, n_video, max_per_type=args.max_eval_pairs)
            if not revisit_vp:
                print("    No valid pre-sampled revisit pairs for this video")
                del vr
                continue
            print(f"    [pre-sampled] Revisit: {len(revisit_vp)}, "
                  f"Baseline: {len(baseline_pairs)}, Short: {len(short_pairs)}", flush=True)
        else:
            min_gap = int(effective_frames * args.min_gap_ratio)
            filtered = []
            for fa, fb, dpos, drot in revisit_pairs:
                va = ern.map_pose_to_video_frame(fa, n_pose, n_video)
                vb = ern.map_pose_to_video_frame(fb, n_pose, n_video)
                if va >= effective_frames or vb >= effective_frames:
                    continue
                if abs(vb - va) >= min_gap:
                    filtered.append((fa, fb, dpos, drot))
            if not filtered:
                print("    No valid revisit video frame pairs")
                del vr
                continue
            sampled = filtered
            if args.max_eval_pairs and len(filtered) > args.max_eval_pairs:
                sampled = random.sample(filtered, args.max_eval_pairs)
            revisit_vp = [(ern.map_pose_to_video_frame(fa, n_pose, n_video),
                           ern.map_pose_to_video_frame(fb, n_pose, n_video))
                          for fa, fb, _, _ in sampled]
            if effective_frames <= 200:
                excl, nbins, gtr = max(5, effective_frames // 50), max(3, effective_frames // 40), 0.3
            else:
                excl, nbins, gtr = max(10, effective_frames // 30), 10, 0.3
            baseline_pairs, _ = ern._sample_gap_matched_temporal_baselines(
                revisit_vp, frame_poses, effective_frames,
                num_temporal_bins=nbins, exclusion_radius=excl,
                gap_tolerance_ratio=gtr, seed=args.seed)
            short_count = min(100, max(20, int(len(revisit_vp) * args.short_multiplier)))
            short_pairs, _ = ern._sample_short_temporal_pairs(
                frame_poses, effective_frames, count=short_count, min_gap=2, seed=args.seed)
            print(f"    Revisit: {len(revisit_vp)}, Baseline: {len(baseline_pairs)}, "
                  f"Short: {len(short_pairs)}", flush=True)

        if want_object:
            object_evaluator.frame_cache.clear()

        eval_kwargs = dict(
            dino_ext=dino_ext, boq_ext=boq_ext, mvpr_ext=mvpr_ext,
            keypoint_matcher=keypoint_matcher, skip_pixel_metrics=skip_pixel_metrics,
            lpips_clip=lpips_clip, object_evaluator=object_evaluator)

        print("    Evaluating revisit pairs...", flush=True)
        revisit_results = _evaluate_pairs(vr, revisit_vp, label="revisit", **eval_kwargs)
        print("    Evaluating baseline pairs...", flush=True)
        baseline_results = _evaluate_pairs(vr, baseline_pairs, label="baseline", **eval_kwargs)
        print("    Evaluating short pairs...", flush=True)
        short_results = _evaluate_pairs(vr, short_pairs, label="short", **eval_kwargs)
        del vr

        revisit_agg = _aggregate(revisit_results)
        baseline_agg = _aggregate(baseline_results)
        short_agg = _aggregate(short_results)
        nmr = _compute_nmr(revisit_agg, baseline_agg, short_agg)

        all_per_video[video_name] = {
            "num_revisit_pairs": len(revisit_vp),
            "num_baseline_pairs": len(baseline_pairs),
            "num_short_pairs": len(short_pairs),
            "revisit": revisit_agg,
            "baseline": baseline_agg,
            "short": short_agg,
            "nmr": nmr,
        }
        for k in nmr:
            print(f"      {k:<28} NMR={nmr[k]:.4f}", flush=True)
        ern._save_results(output_dir, all_per_video)

    # ---- Global summary + final results.json ----------------------------
    g_nmr, g_rev, g_base, g_short = _global_summary(all_per_video)
    final_output = {
        "global_summary": {
            "num_videos": len(all_per_video),
            "families": sorted(families),
            "nmr": g_nmr,
            "revisit_means": g_rev,
            "baseline_means": g_base,
            "short_means": g_short,
        },
        "per_video": all_per_video,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {output_dir / 'results.json'} "
          f"({len(all_per_video)} videos)", flush=True)


if __name__ == "__main__":
    main()
