"""MR5: Gaussian permutation invariance.

Applying one permutation to every per-Gaussian attribute must not change the
image. Scenes with (near-)equal depths are flagged via `depth_tie`, since the
compositing order of tied Gaussians is ambiguous and a difference there is not
necessarily a defect.

Modes:
- "correct": the same permutation is applied to all attributes.
- "misaligned": fault injection; colors are left unpermuted, so attributes no
  longer line up per Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import gsplat
import torch
from metrics import ImageDiff, compare_images
from scene import Camera, GaussianScene


@dataclass
class MR5Result:
    rgb: Optional[ImageDiff]
    alpha: Optional[ImageDiff]
    depth_tie: bool
    error: Optional[str] = None


def has_depth_tie(
    scene: GaussianScene,
    camera: Camera,
    near_plane: float,
    far_plane: float,
    eps: float,
) -> bool:
    z = scene.means @ camera.viewmat[:3, :3].T[:, 2] + camera.viewmat[2, 3]
    z = z[(z > near_plane) & (z < far_plane)].sort().values
    return z.numel() >= 2 and bool((z[1:] - z[:-1]).min() < eps)


def run_mr5(
    scene: GaussianScene,
    camera: Camera,
    seed: int,
    mode: Literal["correct", "misaligned"] = "correct",
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    tie_eps: float = 1e-6,
) -> MR5Result:
    if mode not in ("correct", "misaligned"):
        raise ValueError(f"unknown mode: {mode}")

    perm = torch.randperm(
        scene.n, generator=torch.Generator(device="cpu").manual_seed(seed)
    ).to(scene.means.device)
    colors_p = scene.colors if mode == "misaligned" else scene.colors[perm]

    tie = has_depth_tie(scene, camera, near_plane, far_plane, tie_eps)

    common = dict(
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
        rgb_a, alpha_a, _ = gsplat.rasterization(
            means=scene.means,
            quats=scene.quats,
            scales=scene.scales,
            opacities=scene.opacities,
            colors=scene.colors,
            **common,
        )
        rgb_b, alpha_b, _ = gsplat.rasterization(
            means=scene.means[perm],
            quats=scene.quats[perm],
            scales=scene.scales[perm],
            opacities=scene.opacities[perm],
            colors=colors_p,
            **common,
        )
    except Exception as e:  # noqa: BLE001 - a crash is itself the MR5 finding
        return MR5Result(rgb=None, alpha=None, depth_tie=tie, error=f"{type(e).__name__}: {e}")

    return MR5Result(
        rgb=compare_images(rgb_a, rgb_b),
        alpha=compare_images(alpha_a, alpha_b),
        depth_tie=tie,
    )
