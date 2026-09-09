"""Load pre-sampled revisit/baseline/short pairs shared across all eval types.

sample_pairs.py writes one JSON per unique pose_json under full_eval/pairs/.
Each eval script (NMR / object-consistency / Gemini) accepts --pairs_json
pointing at one such file, so all three score the *same* frame pairs instead
of each re-sampling independently. Pairs are stored in video-frame space
(already mapped through map_pose_to_video_frame at sampling time), so evals
consume them directly with no pose mapping.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

PairList = List[Tuple[int, int]]


def load_pairs_json(pairs_json: str) -> Dict[str, object]:
    """Load a pre-sampled pairs file into revisit/baseline/short tuple lists."""
    data = json.loads(Path(pairs_json).read_text())
    out: Dict[str, object] = {}
    for key in ("revisit", "baseline", "short"):
        out[key] = [(int(a), int(b)) for a, b in data.get(key, [])]
    out["meta"] = data.get("meta", {})
    return out


def select_for_video(
    pairs: Dict[str, object],
    n_video: int,
    max_per_type: int = 0,
) -> Tuple[PairList, PairList, PairList]:
    """Return (revisit, baseline, short) pairs valid for a given video length.

    Any pair whose frame index falls outside [0, n_video) is dropped (the
    pre-sample was computed on a representative video that may differ slightly
    from this one). If max_per_type > 0, a deterministic prefix of each list is
    kept so API-based evals stay within budget while remaining reproducible.
    """
    def _prep(seq: PairList) -> PairList:
        valid = [(a, b) for a, b in seq if 0 <= a < n_video and 0 <= b < n_video]
        if max_per_type and len(valid) > max_per_type:
            valid = valid[:max_per_type]
        return valid

    return (
        _prep(list(pairs.get("revisit", []))),
        _prep(list(pairs.get("baseline", []))),
        _prep(list(pairs.get("short", []))),
    )
