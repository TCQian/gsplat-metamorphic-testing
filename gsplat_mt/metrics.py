"""Numerical comparison metrics for the metamorphic oracle.

Implements D_max and D_MAE from the proposal (main.tex, Section "Metamorphic
Oracle") plus basic NaN/Inf/shape sanity checks so unstable generator
parameters surface immediately instead of silently producing garbage diffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


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
    diff = (a32 - b32).abs()
    return ImageDiff(
        d_max=diff.max().item(),
        d_mae=diff.mean().item(),
        has_nan=bool(torch.isnan(a32).any() or torch.isnan(b32).any()),
        has_inf=bool(torch.isinf(a32).any() or torch.isinf(b32).any()),
        shape=tuple(a.shape),
    )
