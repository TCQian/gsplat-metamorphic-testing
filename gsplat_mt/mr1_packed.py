"""MR1: packed=True vs packed=False rasterization (main.tex, MR1).

Note: gsplat v1.5.3 has a confirmed bug (upstream issue #764) where
`packed=True` combined with `backgrounds` raises an AssertionError in
`rasterize_to_pixels()`. We surface that as a crash-type violation rather than
silently skipping it, since a crash is itself a metamorphic-relation
violation (the two "equivalent" configurations do not both produce a result).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gsplat
import torch
from metrics import ImageDiff, compare_images
from scene import Camera, GaussianScene


@dataclass
class MR1Result:
    rgb: Optional[ImageDiff]
    alpha: Optional[ImageDiff]
    error: Optional[str] = None


def run_mr1(
    scene: GaussianScene,
    camera: Camera,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    use_backgrounds: bool = False,
) -> MR1Result:

    backgrounds = None
    if use_backgrounds:
        backgrounds = torch.rand(
            1, 3, device=scene.means.device, dtype=scene.means.dtype
        )

    common = dict(
        means=scene.means,
        quats=scene.quats,
        scales=scene.scales,
        opacities=scene.opacities,
        colors=scene.colors,
        viewmats=camera.viewmat[None],
        Ks=camera.K[None],
        width=camera.width,
        height=camera.height,
        near_plane=near_plane,
        far_plane=far_plane,
        sh_degree=None,
        backgrounds=backgrounds,
    )

    try:
        rgb_p, alpha_p, _ = gsplat.rasterization(**common, packed=True)
        rgb_u, alpha_u, _ = gsplat.rasterization(**common, packed=False)
    except Exception as e:  # noqa: BLE001 - a crash is itself the MR1 finding
        return MR1Result(rgb=None, alpha=None, error=f"{type(e).__name__}: {e}")

    return MR1Result(
        rgb=compare_images(rgb_p, rgb_u), alpha=compare_images(alpha_p, alpha_u)
    )
