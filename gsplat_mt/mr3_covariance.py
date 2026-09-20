"""MR3: scale+quaternion vs covariance parameterization (main.tex, MR3).

Two covariance construction modes are provided:

- "correct": builds Sigma = R S S^T R^T using gsplat's own public helper
  `gsplat.quat_scale_to_covar_preci`, so the rotation convention (quats are
  (w, x, y, z)) is guaranteed to match what `rasterization()` uses internally
  when given quats/scales directly. Any residual difference here is a
  genuine gsplat-internal numerical discrepancy between its two code paths,
  not a convention mismatch in our test harness.

- "xyzw_bug": deliberately reproduces upstream issue #630
  (https://github.com/nerfstudio-project/gsplat/issues/630), where the
  reporter's `quaternion_to_matrix` unpacks quaternion components as
  `i, j, k, r = unbind(q, dim=-1)`, i.e. treats the scalar component as
  *last* (xyzw), while gsplat's `quats` convention is (w, x, y, z) with the
  scalar *first*. Applying that unbind formula directly to our (w, x, y, z)
  quaternions reproduces the same component mislabeling, isolating exactly
  that one convention bug (same R*S*S^T*R^T formula, wrong rotation matrix).
  This is used as a fault-injection case: it should produce a much larger,
  clearly-detectable MR3 violation than the "correct" baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import gsplat
import torch
from metrics import ImageDiff, compare_images
from scene import Camera, GaussianScene
from torch import Tensor


@dataclass
class MR3Result:
    rgb: Optional[ImageDiff]
    alpha: Optional[ImageDiff]
    error: Optional[str] = None


def _quat_to_rotmat_xyzw_bug(quats: Tensor) -> Tensor:
    """Reproduces the exact rotation-matrix formula from gsplat issue #630,
    applied to quaternions actually stored in gsplat's (w, x, y, z) order.
    """
    eps = 1e-8
    i, j, k, r = torch.unbind(quats, dim=-1)
    two_s = 2.0 / ((quats * quats).sum(dim=-1) + eps)
    R = torch.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        dim=-1,
    )
    return R.reshape(quats.shape[:-1] + (3, 3))


def _covars_xyzw_bug(quats: Tensor, scales: Tensor) -> Tensor:
    R = _quat_to_rotmat_xyzw_bug(quats)  # [N,3,3]
    M = R * scales[..., None, :]  # R @ diag(scales)
    covars = torch.einsum("...ij,...kj->...ik", M, M)  # M @ M^T = R S S^T R^T
    return covars


def run_mr3(
    scene: GaussianScene,
    camera: Camera,
    mode: Literal["correct", "xyzw_bug"] = "correct",
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> MR3Result:

    if mode == "correct":
        covars, _ = gsplat.quat_scale_to_covar_preci(
            scene.quats,
            scene.scales,
            compute_covar=True,
            compute_preci=False,
            triu=False,
        )
    elif mode == "xyzw_bug":
        covars = _covars_xyzw_bug(scene.quats, scene.scales)
    else:
        raise ValueError(f"unknown mode: {mode}")

    common = dict(
        means=scene.means,
        opacities=scene.opacities,
        colors=scene.colors,
        viewmats=camera.viewmat[None],
        Ks=camera.K[None],
        width=camera.width,
        height=camera.height,
        near_plane=near_plane,
        far_plane=far_plane,
        sh_degree=None,
        packed=False,
    )

    try:
        rgb_sq, alpha_sq, _ = gsplat.rasterization(
            quats=scene.quats, scales=scene.scales, covars=None, **common
        )
        # quats/scales are ignored internally when covars is given, but the
        # positional/keyword arguments are still required by the signature.
        rgb_cov, alpha_cov, _ = gsplat.rasterization(
            quats=scene.quats, scales=scene.scales, covars=covars, **common
        )
    except Exception as e:  # noqa: BLE001 - a crash is itself the MR3 finding
        return MR3Result(rgb=None, alpha=None, error=f"{type(e).__name__}: {e}")

    return MR3Result(
        rgb=compare_images(rgb_sq, rgb_cov), alpha=compare_images(alpha_sq, alpha_cov)
    )
