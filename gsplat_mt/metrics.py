"""Numerical comparison metrics and pass/violation verdicts for the metamorphic oracle.

D_max / D_MAE plus NaN/Inf/shape checks. A NaN/Inf mismatch between the two
images counts as an infinite difference rather than being dropped by max().
Tolerances come from the preliminary study's measured noise floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


# Max absolute pixel difference tolerated per relation. MR1 packed/unpacked
# showed ~1e-7 float noise; MR2 and MR3 (correct) were bit-identical. MR4
# rescales every float, so it gets MR1's tolerance; MR5 should be exact away
# from depth ties. MR4/MR5 values are initial and need calibrating on a run.
TOLERANCES = {"mr1": 1e-4, "mr2": 0.0, "mr3": 0.0, "mr4": 1e-4, "mr5": 0.0}

# A tolerance violation touching at most this fraction of pixels is relabeled
# `isolated_discontinuity` for the float-noise relations (MR1, MR4): one pixel
# crossing a rasterizer cutoff (e.g. alpha < 1/255) is not an image-wide error.
ISOLATED_PIXEL_FRAC = 5e-4

# ...and its difference must also be no larger than what one cutoff flip can
# produce: a Gaussian with alpha ~1/255 (gsplat's ALPHA_THRESHOLD) blended in
# one render and skipped in the other changes a pixel by at most
# color x alpha x T <= 1/255 = 3.9e-3 (colors are in [0, 1]).
ISOLATED_MAX_D_MAX = 4e-3


@dataclass
class ImageDiff:
    d_max: float
    d_mae: float
    has_nan: bool
    has_inf: bool
    shape: Tuple[int, ...]
    n_px_over: int  # pixels whose max channel difference exceeds the tol given to compare_images
    n_pixels: int

    def as_row(self, prefix: str) -> dict:
        return {
            f"{prefix}_d_max": self.d_max,
            f"{prefix}_d_mae": self.d_mae,
            f"{prefix}_has_nan": self.has_nan,
            f"{prefix}_has_inf": self.has_inf,
            f"{prefix}_n_px_over": self.n_px_over,
        }


def compare_images(a: Tensor, b: Tensor, tol: float = 0.0) -> ImageDiff:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    a32 = a.detach().float()
    b32 = b.detach().float()
    same = (a32 == b32) | (torch.isnan(a32) & torch.isnan(b32))
    diff = torch.where(same, torch.zeros_like(a32), (a32 - b32).abs())
    diff = torch.where(torch.isnan(diff), torch.full_like(diff, float("inf")), diff)
    return ImageDiff(
        d_max=diff.max().item(),
        d_mae=diff.mean().item(),
        has_nan=bool(torch.isnan(a32).any() or torch.isnan(b32).any()),
        has_inf=bool(torch.isinf(a32).any() or torch.isinf(b32).any()),
        shape=tuple(a.shape),
        n_px_over=int((diff.amax(dim=-1) > tol).sum().item()),
        n_pixels=diff[..., 0].numel(),
    )


def verdict(
    diffs: Tuple[Optional[ImageDiff], ...],
    error: Optional[str],
    tol: float,
    allow_isolated: bool = False,
) -> str:
    """crash | violation | isolated_discontinuity | invalid_output | pass.

    isolated_discontinuity (only if `allow_isolated`): every image over `tol`
    has at most ISOLATED_PIXEL_FRAC of its pixels over `tol` and a D_max of at most
    ISOLATED_MAX_D_MAX; the diffs must have been built with compare_images(..., tol=tol).
    invalid_output: both configurations agree within `tol` but emit NaN/Inf.
    """
    if error is not None:
        return "crash"
    diffs = tuple(d for d in diffs if d is not None)
    over = [d for d in diffs if d.d_max > tol]
    if over:
        if allow_isolated and all(
            d.n_px_over <= int(ISOLATED_PIXEL_FRAC * d.n_pixels) and d.d_max <= ISOLATED_MAX_D_MAX
            for d in over
        ):
            return "isolated_discontinuity"
        return "violation"
    if any(d.has_nan or d.has_inf for d in diffs):
        return "invalid_output"
    return "pass"


def is_power_of_two(k: float) -> bool:
    """True if k = 2^n exactly, so multiplying by k introduces no float rounding."""
    m, _ = math.frexp(k)
    return m == 0.5
