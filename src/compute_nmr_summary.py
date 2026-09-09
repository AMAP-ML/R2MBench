#!/usr/bin/env python3
"""Aggregate sharded R2M-Bench results into a template-balanced NMR summary.

Reads the `full_eval` result layout produced by `run_full_eval.sh`:

    results/nmr/<family>/<task>/shard_*/results.json        (or results_partial.json)
    results/nmr/persistent_state/results_gemini_nmr_<task>.json

where <family> is one of appearance / scene_identity / geometric / object / unified
and <task> is `<model>_<trajectory>_30s` (e.g. `modelA_1400_DDDAAA_30s`).

Per-video revisit / baseline / short means are merged across families by video id,
then reduced with the video-balanced NMR protocol (see the emitted markdown header
for the exact formulas). Output mirrors `benchv2_result_nmr_30s_summary_final.md`:
Video Counts, Template-Balanced NMR, and a Per-Trajectory NMR table per trajectory.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


# ---- Metric definitions (order = display order) --------------------------

METRICS = [
    "psnr",
    "ssim",
    "lpips",
    "dino_similarity",
    "boq_similarity",
    "mvpr_similarity",
    "match_ratio",
    "ransac_inlier_ratio",
    "state_score",
    "object_appearance_persistence",
    "object_semantic_persistence",
]

LOWER_IS_BETTER = {"lpips"}

METRIC_LABELS = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "lpips": "LPIPS",
    "dino_similarity": "DINOv2",
    "boq_similarity": "BoQ",
    "mvpr_similarity": "MutualVPR",
    "match_ratio": "Match Ratio",
    "ransac_inlier_ratio": "RANSAC Inlier",
    "state_score": "Gemini State",
    "object_appearance_persistence": "App. Persist",
    "object_semantic_persistence": "Sem. Persist",
}

FAMILIES = {
    "Appearance": ["psnr", "ssim", "lpips"],
    "Scene ID": ["dino_similarity", "boq_similarity", "mvpr_similarity"],
    "Local Geo.": ["match_ratio", "ransac_inlier_ratio"],
    "State": ["state_score"],
    "Object": ["object_appearance_persistence", "object_semantic_persistence"],
}

# Which metrics belong to each count bucket shown in the "Video Counts" table.
VISUAL_METRICS = {
    "psnr", "ssim", "lpips",
    "dino_similarity", "boq_similarity", "mvpr_similarity",
    "match_ratio", "ransac_inlier_ratio",
}
OBJECT_METRICS = {"object_appearance_persistence", "object_semantic_persistence"}

# Visual result families to scan (unified carries every visual metric per video).
VISUAL_FAMILY_DIRS = ["appearance", "scene_identity", "geometric", "object", "unified"]
STATE_FAMILY_DIR = "persistent_state"

TEMPLATE_ORDER = ["DDDAAA", "WSLRRLL", "revisit_loop"]
TEMPLATE_LABELS = {
    "DDDAAA": "DDDAAA",
    "WSLRRLL": "WSLRRLL",
    "revisit_loop": "Revisit Loop",
}
# Longest suffix first so `revisit_loop` wins over a bare template check.
TEMPLATE_SUFFIXES = [
    ("revisit_loop", "_revisit_loop_30s"),
    ("DDDAAA", "_dddaaa_30s"),
    ("WSLRRLL", "_wslrrll_30s"),
]


# ---- Task-name parsing ---------------------------------------------------

def parse_task_name(name: str) -> Optional[tuple[str, str]]:
    """Split `<model>_<trajectory>_30s` into (model, template).

    Returns None if no known trajectory token is present.
    """
    lower = name.lower()
    for template, suffix in TEMPLATE_SUFFIXES:
        if lower.endswith(suffix):
            return name[: len(name) - len(suffix)], template
    # Fallback: token appears but not as a clean suffix.
    for template, suffix in TEMPLATE_SUFFIXES:
        token = suffix.strip("_").rsplit("_30s", 1)[0]
        if token in lower:
            model = lower.replace(token, "").replace("_30s", "").strip("_")
            return (model or name), template
    return None


def parse_gemini_filename(path: Path) -> Optional[tuple[str, str]]:
    stem = path.stem
    prefix = "results_gemini_nmr_"
    if not stem.startswith(prefix):
        return None
    return parse_task_name(stem[len(prefix):])


# ---- Small helpers -------------------------------------------------------

def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def load_shard(shard_dir: Path) -> dict[str, Any]:
    """Prefer the finished results.json, fall back to results_partial.json."""
    for name in ("results.json", "results_partial.json"):
        path = shard_dir / name
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
    return {}


# ---- Collection ----------------------------------------------------------

def _blank_video() -> dict[str, dict[str, Any]]:
    return {"revisit": {}, "baseline": {}, "short": {}}


def collect_visual(results_dir: Path) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    """Merge per-video revisit/baseline/short metric dicts across visual families.

    grouped[(model, template)][video_id] = {revisit: {...}, baseline: {...}, short: {...}}
    """
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for family in VISUAL_FAMILY_DIRS:
        family_dir = results_dir / family
        if not family_dir.is_dir():
            continue
        for task_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
            parsed = parse_task_name(task_dir.name)
            if parsed is None:
                continue
            key = parsed
            for shard_dir in sorted(task_dir.glob("shard_*")):
                data = load_shard(shard_dir)
                for vid, entry in data.get("per_video", {}).items():
                    video = grouped[key].setdefault(vid, _blank_video())
                    for split in ("revisit", "baseline", "short"):
                        for metric, stats in entry.get(split, {}).items():
                            video[split][metric] = stats
    return grouped


def collect_state(
    results_dir: Path,
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]],
) -> None:
    """Fold Gemini persistent-state scores in as a `state_score` metric."""
    state_dir = results_dir / STATE_FAMILY_DIR
    if not state_dir.is_dir():
        return
    for path in sorted(state_dir.glob("results_gemini_nmr_*.json")):
        parsed = parse_gemini_filename(path)
        if parsed is None:
            continue
        key = parsed
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for vid, video in data.get("videos", {}).items():
            rev = [float(x) for x in video.get("revisit_scores", [])]
            base = [float(x) for x in video.get("baseline_scores", [])]
            short = [float(x) for x in video.get("short_scores", [])]
            if rev and base and short:
                r, b, s = mean(rev), mean(base), mean(short)
            else:
                block = video.get("nmr", {})
                r = block.get("mean_revisit")
                b = block.get("mean_baseline")
                s = block.get("mean_short")
            if r is None or b is None or s is None:
                continue
            entry = grouped[key].setdefault(vid, _blank_video())
            entry["revisit"]["state_score"] = {"mean": r}
            entry["baseline"]["state_score"] = {"mean": b}
            entry["short"]["state_score"] = {"mean": s}


# ---- Aggregation ---------------------------------------------------------

def video_mg_dr(video: dict[str, Any], metric: str) -> Optional[tuple[float, float]]:
    try:
        rev = float(video["revisit"][metric]["mean"])
        base = float(video["baseline"][metric]["mean"])
        short = float(video["short"][metric]["mean"])
    except (KeyError, TypeError, ValueError):
        return None
    if metric in LOWER_IS_BETTER:
        return base - rev, base - short
    return rev - base, short - base


def aggregate_metric(
    videos: list[dict[str, Any]], metric: str, eps: float
) -> dict[str, float]:
    pairs = [p for p in (video_mg_dr(v, metric) for v in videos) if p is not None]
    valid = [(mg, dr) for mg, dr in pairs if dr > eps]
    if not valid:
        return {"n": 0, "mg": math.nan, "dr": math.nan, "nmr": math.nan}
    mg = mean([m for m, _ in valid])
    dr = mean([d for _, d in valid])
    return {"n": len(valid), "mg": mg, "dr": dr, "nmr": mg / (dr + eps)}


def aggregate_all(
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]], eps: float
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(dict)
    for (model, template), videos in grouped.items():
        vlist = list(videos.values())
        summary[model][template] = {
            metric: aggregate_metric(vlist, metric, eps) for metric in METRICS
        }
    return summary


def metric_stats(
    summary: dict[str, dict[str, dict[str, dict[str, float]]]],
    model: str,
    template: Optional[str],
    metric: str,
) -> dict[str, float]:
    if template is not None:
        return summary[model].get(template, {}).get(
            metric, {"mg": math.nan, "dr": math.nan, "nmr": math.nan}
        )
    cells = [
        summary[model][t][metric]
        for t in TEMPLATE_ORDER
        if t in summary[model] and not math.isnan(summary[model][t][metric]["nmr"])
    ]
    if not cells:
        return {"mg": math.nan, "dr": math.nan, "nmr": math.nan}
    return {
        "mg": mean([c["mg"] for c in cells]),
        "dr": mean([c["dr"] for c in cells]),
        "nmr": mean([c["nmr"] for c in cells]),
    }


def overall_value(
    summary: dict[str, dict[str, dict[str, dict[str, float]]]],
    model: str,
    template: Optional[str],
) -> float:
    family_values = []
    for metrics in FAMILIES.values():
        values = [metric_stats(summary, model, template, m)["nmr"] for m in metrics]
        valid_values = [v for v in values if not math.isnan(v)]
        if valid_values:
            family_values.append(mean(valid_values))
    return mean(family_values) if family_values else math.nan


# ---- Rendering -----------------------------------------------------------

def fmt(value: float, digits: int = 3) -> str:
    return "--" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def count_cell(videos: dict[str, dict[str, Any]], metrics: set[str]) -> int:
    return sum(
        1 for v in videos.values()
        if any(m in v.get("revisit", {}) for m in metrics)
    )


def html_nmr_table(
    summary: dict[str, dict[str, dict[str, dict[str, float]]]],
    models: list[str],
    template: Optional[str],
) -> str:
    lines = ["<table>", "<thead>", "<tr>", '<th rowspan="3">Method</th>']
    for family, metrics in FAMILIES.items():
        lines.append(f'<th colspan="{2 * len(metrics)}">{family}</th>')
    lines.append('<th rowspan="3">Overall NMR</th>')
    lines.extend(["</tr>", "<tr>"])
    for metrics in FAMILIES.values():
        for metric in metrics:
            lines.append(f'<th colspan="2">{METRIC_LABELS[metric]}</th>')
    lines.extend(["</tr>", "<tr>"])
    for metrics in FAMILIES.values():
        for _ in metrics:
            lines.append("<th>Gain</th>")
            lines.append("<th>NMR</th>")
    lines.extend(["</tr>", "</thead>", "<tbody>"])

    for model in models:
        if template is not None and template not in summary[model]:
            continue
        lines.append("<tr>")
        lines.append(f"<td>{model}</td>")
        for metrics in FAMILIES.values():
            for metric in metrics:
                item = metric_stats(summary, model, template, metric)
                if math.isnan(item["nmr"]):
                    lines.append("<td>--</td>")
                    lines.append("<td>--</td>")
                else:
                    lines.append(f"<td>{fmt(item['mg'])}</td>")
                    lines.append(f"<td>{fmt(item['nmr'])}</td>")
        lines.append(f"<td>{fmt(overall_value(summary, model, template))}</td>")
        lines.append("</tr>")

    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def render_markdown(
    summary: dict[str, dict[str, dict[str, dict[str, float]]]],
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]],
    results_dir: Path,
    eps: float,
) -> str:
    models = sorted(summary)

    lines = [
        "# NMR Results",
        "",
        f"Source directory: `{results_dir}`",
        "",
        "Computation follows the video-balanced protocol:",
        "",
        "- For each video and metric, compute Rev/Base/Short means from pair summaries.",
        "- Higher-is-better metrics use `MG_v = Rev_v - Base_v` and `DR_v = Short_v - Base_v`.",
        "- Lower-is-better metrics use `MG_v = Base_v - Rev_v` and `DR_v = Base_v - Short_v`.",
        "- Video-metric cases with `DR_v <= eps` are invalid and excluded from that metric's aggregation.",
        "- For each trajectory and metric, average valid video-level MG and DR first, then compute `NMR = mean(MG_v) / (mean(DR_v) + eps)`.",
        "- The final average is template-balanced: arithmetic mean of the per-trajectory NMR values.",
        "- `Overall NMR` first averages metric-level NMR values within each family, then averages the available family means.",
        "",
        f"`eps = {eps:g}`. A metric is reported as `--` only when no valid video remains.",
        "",
        "## Video Counts",
        "",
        "Each cell is `visual metrics videos / Gemini state videos / object-consistency videos`.",
        "",
    ]

    count_rows = []
    for model in models:
        row = [model]
        for template in TEMPLATE_ORDER:
            videos = grouped.get((model, template), {})
            visual_n = count_cell(videos, VISUAL_METRICS)
            state_n = count_cell(videos, {"state_score"})
            object_n = count_cell(videos, OBJECT_METRICS)
            row.append(f"{visual_n} / {state_n} / {object_n}")
        count_rows.append(row)
    lines.append(markdown_table(
        ["Method", *[TEMPLATE_LABELS[t] for t in TEMPLATE_ORDER]], count_rows
    ))

    lines.extend(["", "## Template-Balanced NMR", ""])
    lines.append(html_nmr_table(summary, models, None))

    lines.extend(["", "## Per-Trajectory NMR", ""])
    for template in TEMPLATE_ORDER:
        lines.extend(["", f"### {TEMPLATE_LABELS[template]}", ""])
        lines.append(html_nmr_table(summary, models, template))

    return "\n".join(lines) + "\n"


def filter_methods(
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]],
    patterns: Optional[list[str]],
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    """Keep only (model, template) entries whose model matches a glob pattern."""
    if not patterns:
        return grouped
    return {
        (model, template): videos
        for (model, template), videos in grouped.items()
        if any(fnmatch.fnmatch(model, pat) for pat in patterns)
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root / "results" / "nmr",
        help="Root of the NMR result tree (default: results/nmr).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results" / "nmr_30s_summary.md",
        help="Markdown file to write.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated glob patterns on the model name; only matching "
             "methods are aggregated (e.g. 'modelA_1400' or 'modelA_*'). "
             "Default: all discovered methods.",
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    grouped = collect_visual(args.results_dir)
    collect_state(args.results_dir, grouped)
    if not grouped:
        raise SystemExit(f"No results found under {args.results_dir}")

    if args.methods:
        available = sorted({model for model, _ in grouped})
        patterns = [p.strip() for p in args.methods.split(",") if p.strip()]
        grouped = filter_methods(grouped, patterns)
        if not grouped:
            raise SystemExit(
                f"No methods matched {patterns}. Available methods: {available}"
            )

    summary = aggregate_all(grouped, args.eps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary, grouped, args.results_dir, args.eps))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
