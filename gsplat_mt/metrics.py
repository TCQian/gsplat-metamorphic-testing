"""Numerical comparison metrics and pass/violation verdicts for the metamorphic oracle.

D_max / D_MAE plus NaN/Inf/shape checks. A NaN/Inf mismatch between the two
images counts as an infinite difference rather than being dropped by max().
Tolerances come from the preliminary study's measured noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


# Max absolute pixel difference tolerated per relation. MR1 packed/unpacked
# showed ~1e-7 float noise; MR2 and MR3 (correct) were bit-identical.
TOLERANCES = {"mr1": 1e-4, "mr2": 0.0, "mr3": 0.0}


@dataclass
class ImageDiff:
    d_max: float
    d_mae: float
    has_nan: bool
    has_inf: bool
    shape: Tuple[int, ...]

    def as_row(self, prefix: str) -> dict:
        return {
            f"{prefix}_d_max": self.d_max,
            f"{prefix}_d_mae": self.d_mae,
            f"{prefix}_has_nan": self.has_nan,
            f"{prefix}_has_inf": self.has_inf,
        }


def compare_images(a: Tensor, b: Tensor) -> ImageDiff:
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
    )


def verdict(
    diffs: Tuple[Optional[ImageDiff], ...], error: Optional[str], tol: float
) -> str:
    """crash | violation | invalid_output | pass.

    invalid_output: both configurations agree within `tol` but emit NaN/Inf.
    """
    if error is not None:
        return "crash"
    diffs = tuple(d for d in diffs if d is not None)
    if any(d.d_max > tol for d in diffs):
        return "violation"
    if any(d.has_nan or d.has_inf for d in diffs):
        return "invalid_output"
    return "pass"
