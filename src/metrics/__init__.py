"""Pairwise consistency metric families for MemoryGain / NMR evaluation.

Each of the five metric families defined in the paper lives in its own module:

    appearance_fidelity      - PSNR, SSIM, LPIPS
    scene_identity           - DINOv2, BoQ, MutualVPR global-descriptor similarity
    object_identity          - GroundingDINO + SAM2 + DINOv2/CLIP object persistence
    geometric_correspondence - SuperPoint + LightGlue matching, RANSAC two-view geometry
    persistent_state         - VLM structured-rubric pairwise state consistency

Heavy third-party model deps (groundingdino, sam2, clip, lightglue, boq,
cosplace) are imported lazily inside the relevant classes/functions, so
importing this package is cheap and does not require every backend to be
installed.
"""

from .appearance_fidelity import compute_psnr_ssim, LpipsClipComputer
from .scene_identity import (
    DINOv2FeatureExtractor,
    BoQFeatureExtractor,
    MutualVPRFeatureExtractor,
)
from .geometric_correspondence import KeypointMatcher, compute_two_view_geometry
from .object_identity import (
    ObjectInstance,
    CLIPFeatureExtractor,
    GroundedSamObjectEvaluator,
    DEFAULT_TEXT_PROMPT,
)
from .persistent_state import (
    EVAL_SYSTEM_PROMPT,
    EVAL_USER_PROMPT,
    parse_score_response,
    extract_frame_as_pil,
    build_pair_images,
    evaluate_frame_pairs_gemini,
    compute_gemini_nmr,
)

__all__ = [
    # Appearance Fidelity
    "compute_psnr_ssim",
    "LpipsClipComputer",
    # Scene Identity Preservation
    "DINOv2FeatureExtractor",
    "BoQFeatureExtractor",
    "MutualVPRFeatureExtractor",
    # Local Geometric Correspondence
    "KeypointMatcher",
    "compute_two_view_geometry",
    # Object Identity
    "ObjectInstance",
    "CLIPFeatureExtractor",
    "GroundedSamObjectEvaluator",
    "DEFAULT_TEXT_PROMPT",
    # Persistent State Reasoning
    "EVAL_SYSTEM_PROMPT",
    "EVAL_USER_PROMPT",
    "parse_score_response",
    "extract_frame_as_pil",
    "build_pair_images",
    "evaluate_frame_pairs_gemini",
    "compute_gemini_nmr",
]
