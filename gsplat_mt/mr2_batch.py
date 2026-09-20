"""MR2: batched vs individual camera rendering (main.tex, MR2).

All cameras passed to `run_mr2` must share the same (width, height) so the
batched call's stacked viewmats/Ks/output tensor are well-formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import gsplat
import torch
from metrics import ImageDiff, compare_images
from scene import Camera, GaussianScene


@dataclass
class MR2Result:
    per_camera: List[Optional[ImageDiff]]
    error: Optional[str] = None


def run_mr2(
    scene: GaussianScene,
    cameras: List[Camera],
    packed: bool = False,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> MR2Result:

    widths = {c.width for c in cameras}
    heights = {c.height for c in cameras}
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError("run_mr2 requires all cameras to share the same resolution")
    width, height = cameras[0].width, cameras[0].height

    viewmats = torch.stack([c.viewmat for c in cameras], dim=0)  # [C,4,4]
    Ks = torch.stack([c.K for c in cameras], dim=0)  # [C,3,3]

    common = dict(
        means=scene.means,
        quats=scene.quats,
        scales=scene.scales,
        opacities=scene.opacities,
        colors=scene.colors,
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
        sh_degree=None,
        packed=packed,
    )

    try:
        rgb_batched, _, _ = gsplat.rasterization(viewmats=viewmats, Ks=Ks, **common)
        per_camera = []
        for i, cam in enumerate(cameras):
            rgb_i, _, _ = gsplat.rasterization(
                viewmats=cam.viewmat[None], Ks=cam.K[None], **common
            )
            per_camera.append(compare_images(rgb_batched[i : i + 1], rgb_i))
    except Exception as e:  # noqa: BLE001 - a crash is itself the MR2 finding
        return MR2Result(per_camera=[], error=f"{type(e).__name__}: {e}")

    return MR2Result(per_camera=per_camera)
