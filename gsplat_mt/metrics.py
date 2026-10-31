"""Numerical comparison metrics and pass/violation verdicts for the metamorphic oracle.

D_max / D_MAE plus NaN/Inf/shape checks. A NaN/Inf mismatch between the two
images counts as an infinite difference rather than being dropped by max().
Tolerances come from the preliminary study's measured noise floor.

Also computes D_SSIM (`ssim`), a windowed structural-similarity score: unlike
D_max/D_MAE it is local and averaged, so it stays near 1.0 for the single-pixel
float noise this project already tolerates while dropping for changes spread
over a region, even at the same D_max. Diagnostic only for now -- recorded on
every ImageDiff but not yet consulted by `verdict()`.
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

# SSIM window (Wang et al. 2004's standard 11x11 Gaussian, sigma 1.5); also
# gsplat/3DGS's own domain-standard quality metric alongside PSNR/LPIPS.
# Assumes inputs are in [0, 1] (rendered colors and alpha both are).
_SSIM_WINDOW = 11
_SSIM_SIGMA = 1.5
_SSIM_DATA_RANGE = 1.0


@dataclass
class ImageDiff:
    d_max: float
    d_mae: float
    d_ssim: float
    has_nan: bool
    has_inf: bool
    shape: Tuple[int, ...]
    n_px_over: int  # pixels whose max channel difference exceeds the tol given to compare_images
    n_pixels: int

    def as_row(self, prefix: str) -> dict:
        return {
            f"{prefix}_d_max": self.d_max,
            f"{prefix}_d_mae": self.d_mae,
            f"{prefix}_d_ssim": self.d_ssim,
            f"{prefix}_has_nan": self.has_nan,
            f"{prefix}_has_inf": self.has_inf,
            f"{prefix}_n_px_over": self.n_px_over,
        }


def _gaussian_kernel(window: int, sigma: float, device, dtype) -> Tensor:
    coords = torch.arange(window, device=device, dtype=dtype) - (window - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def _ssim_global(a: Tensor, b: Tensor, data_range: float) -> float:
    """Unwindowed SSIM over the whole image; used only when it is smaller than the window."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(unbiased=False), b.var(unbiased=False)
    cov_ab = ((a - mu_a) * (b - mu_b)).mean()
    val = ((2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )
    return val.item()


def ssim(a: Tensor, b: Tensor, data_range: float = _SSIM_DATA_RANGE) -> float:
    """Mean structural similarity (Wang et al. 2004) between (..., H, W, C) images.

    1.0 is identical; it falls off with locally-windowed differences in
    luminance, contrast and structure rather than raw per-pixel magnitude. See
    the module docstring for why that makes it a useful complement to
    D_max/D_MAE. NaN/Inf in either image propagate into a NaN score, same as
    the `has_nan`/`has_inf` flags already carry that information.
    """
    # float64: a float32 mean near 1.0 resolves only ~6e-8, about the size of 1 - SSIM for a
    # single cutoff-flip pixel (~1e-7), so float32 would hide the isolated cases.
    a32, b32 = a.detach().double(), b.detach().double()
    *_, h, w, c = a32.shape
    a32 = a32.reshape(-1, h, w, c).permute(0, 3, 1, 2)  # [N,C,H,W]
    b32 = b32.reshape(-1, h, w, c).permute(0, 3, 1, 2)

    window = min(_SSIM_WINDOW, h, w)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return _ssim_global(a32, b32, data_range)

    kernel_1d = _gaussian_kernel(window, _SSIM_SIGMA, a32.device, a32.dtype)
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(c, 1, window, window).contiguous()
    pad = window // 2

    def blur(x: Tensor) -> Tensor:
        return torch.nn.functional.conv2d(x, kernel, padding=pad, groups=c)

    mu_a, mu_b = blur(a32), blur(b32)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    var_a = blur(a32 * a32) - mu_a2
    var_b = blur(b32 * b32) - mu_b2
    cov_ab = blur(a32 * b32) - mu_ab

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_ab + c1) * (2 * cov_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
    )
    return ssim_map.mean().item()


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
        d_ssim=ssim(a32, b32),
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
