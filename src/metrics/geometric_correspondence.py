import os
import sys
from typing import Optional

import numpy as np
import torch

def _prepend_path_from_env(name: str) -> None:
    repo_path = os.environ.get(name)
    if not repo_path:
        raise EnvironmentError(f"{name} is not configured. Please set it in run_config.conf.")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

_prepend_path_from_env("LIGHTGLUE_ROOT")

class KeypointMatcher:
    """SuperPoint + LightGlue keypoint matching."""

    def __init__(self, device: str = "cuda:0", max_keypoints: int = 1024):
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import numpy_image_to_torch

        self.numpy_image_to_torch = numpy_image_to_torch
        self.device = device
        self.superpoint = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.lightglue = LightGlue(features="superpoint").eval().to(device)
        print(f"  SuperPoint + LightGlue loaded on {device}.")

    @torch.no_grad()
    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> dict:
        """Compute keypoint matches between two images.

        Returns dict with:
            num_keypoints_a, num_keypoints_b: detected keypoints per image
            num_matches: number of matched keypoint pairs
            match_ratio: num_matches / min(num_keypoints_a, num_keypoints_b)
            matched_kp_a: (N, 2) numpy array of matched keypoints in image A
            matched_kp_b: (N, 2) numpy array of matched keypoints in image B
        """
        tensor_a = self.numpy_image_to_torch(img_a).to(self.device)
        tensor_b = self.numpy_image_to_torch(img_b).to(self.device)

        feats_a = self.superpoint.extract(tensor_a)
        feats_b = self.superpoint.extract(tensor_b)

        num_kp_a = int(feats_a["keypoints"].shape[1])
        num_kp_b = int(feats_b["keypoints"].shape[1])

        matches_result = self.lightglue({"image0": feats_a, "image1": feats_b})
        matches0 = matches_result["matches0"][0]  # (num_kp_a,)
        valid_mask = matches0 > -1
        num_matches = int(valid_mask.sum().item())
        min_kp = min(num_kp_a, num_kp_b)
        match_ratio = round(num_matches / max(min_kp, 1), 4)

        # Extract matched keypoint coordinates
        kp_a = feats_a["keypoints"][0]  # (num_kp_a, 2)
        kp_b = feats_b["keypoints"][0]  # (num_kp_b, 2)
        matched_indices_a = torch.where(valid_mask)[0]
        matched_indices_b = matches0[valid_mask].long()
        matched_kp_a = kp_a[matched_indices_a].cpu().numpy() if num_matches > 0 else np.empty((0, 2))
        matched_kp_b = kp_b[matched_indices_b].cpu().numpy() if num_matches > 0 else np.empty((0, 2))

        return {
            "num_keypoints_a": num_kp_a,
            "num_keypoints_b": num_kp_b,
            "num_matches": num_matches,
            "match_ratio": match_ratio,
            "matched_kp_a": matched_kp_a,
            "matched_kp_b": matched_kp_b,
        }


def compute_two_view_geometry(
    matched_kp_a: Optional[np.ndarray],
    matched_kp_b: Optional[np.ndarray],
) -> dict:
    """Robust two-view geometry from matched keypoints (RANSAC).

    Estimates the fundamental matrix via RANSAC and reports the inlier ratio.
    Requires >= 8 matches; degrades gracefully to a zero-inlier default.

    Args:
        matched_kp_a, matched_kp_b: (N, 2) arrays of matched keypoints.

    Returns dict with: ransac_inlier_ratio, num_ransac_inliers.
    """
    zero = {"ransac_inlier_ratio": 0.0, "num_ransac_inliers": 0}

    if matched_kp_a is not None and len(matched_kp_a) >= 8:
        try:
            import cv2
            fund_mat, mask = cv2.findFundamentalMat(
                matched_kp_a.astype(np.float64),
                matched_kp_b.astype(np.float64),
                cv2.FM_RANSAC, 3.0, 0.99,
            )
            if fund_mat is None or mask is None:
                return dict(zero)

            inlier_mask = mask.ravel().astype(bool)
            num_inliers = int(inlier_mask.sum())
            return {
                "ransac_inlier_ratio": round(num_inliers / len(matched_kp_a), 6),
                "num_ransac_inliers": num_inliers,
            }
        except Exception:
            return dict(zero)
    else:
        return dict(zero)
