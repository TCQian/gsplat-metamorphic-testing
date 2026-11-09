"""MR4: uniform world-scale invariance.

Multiplying Gaussian means and scales, camera translation, and near/far planes
by k > 0 (intrinsics, rotations, colors, opacities unchanged) must not change
the image.

Modes:
- "correct": full joint transform T_k.
- "no_translation": fault injection; the camera translation is left unscaled,
  so the camera no longer sits at the same relative position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import gsplat
from metrics import TOLERANCES, ImageDiff, compare_images
from scene import Camera, GaussianScene, stack_cameras


@dataclass
class MR4Result:
    rgb: Optional[ImageDiff]
    alpha: Optional[ImageDiff]
    error: Optional[str] = None


def run_mr4(
    scene: GaussianScene,
    cameras: Union[Camera, Sequence[Camera]],
    k: float,
    mode: Literal["correct", "no_translation"] = "correct",
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> MR4Result:
    if mode not in ("correct", "no_translation"):
        raise ValueError(f"unknown mode: {mode}")

    viewmats, Ks, width, height = stack_cameras(cameras)
    viewmats_k = viewmats.clone()
    if mode == "correct":
        viewmats_k[:, :3, 3] *= k  # every camera's translation

    common = dict(
        quats=scene.quats,
        opacities=scene.opacities,
        colors=scene.colors,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=None,
        packed=False,
    )

    try:
        rgb_a, alpha_a, _ = gsplat.rasterization(
            means=scene.means,
            scales=scene.scales,
            viewmats=viewmats,
            near_plane=near_plane,
            far_plane=far_plane,
            **common,
        )
        rgb_b, alpha_b, _ = gsplat.rasterization(
            means=scene.means * k,
            scales=scene.scales * k,
            viewmats=viewmats_k,
            near_plane=near_plane * k,
            far_plane=far_plane * k,
            **common,
        )
    except Exception as e:  # noqa: BLE001 - a crash is itself the MR4 finding
        return MR4Result(rgb=None, alpha=None, error=f"{type(e).__name__}: {e}")

    return MR4Result(
        rgb=compare_images(rgb_a, rgb_b, TOLERANCES["mr4"]),
        alpha=compare_images(alpha_a, alpha_b, TOLERANCES["mr4"]),
    )
