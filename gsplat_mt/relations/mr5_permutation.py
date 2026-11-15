"""MR5: Gaussian permutation invariance.

Applying one permutation to every per-Gaussian attribute must not change the
image. Scenes with (near-)equal depths are flagged via `depth_tie`, since the
compositing order of tied Gaussians is ambiguous and a difference there is not
necessarily a defect.

The permutation is a derangement (no Gaussian keeps its index), drawn from the
scene's seed. A random shuffle can leave every visible Gaussian in place, and
then the planted fault below cannot change any pixel; a derangement moves every
Gaussian. With one Gaussian no shuffle can move anything.

Modes:
- "correct": the same permutation is applied to all attributes.
- "misaligned": fault injection; colors are left unpermuted, so attributes no
  longer line up per Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import gsplat
import torch
from torch import Tensor
from gsplat_mt.metrics import ImageDiff, compare_images
from gsplat_mt.scene import Camera, GaussianScene, as_camera_list, stack_cameras


@dataclass
class MR5Result:
    rgb: Optional[ImageDiff]
    alpha: Optional[ImageDiff]
    depth_tie: bool
    error: Optional[str] = None


def random_shuffle(n: int, seed: int) -> Tensor:
    """A uniformly random permutation of range(n) (may leave Gaussians in place)."""
    return torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(seed))


def derangement(n: int, seed: int) -> Tensor:
    """A permutation of range(n) with no fixed point for n >= 2: one n-cycle through a random order."""
    q = random_shuffle(n, seed)
    perm = torch.empty(n, dtype=torch.long)
    perm[q] = q.roll(-1)  # perm[q[j]] = q[j + 1]
    return perm


def has_depth_tie(
    scene: GaussianScene,
    cameras: Union[Camera, Sequence[Camera]],
    near_plane: float,
    far_plane: float,
    eps: float,
) -> bool:
    """True if, in any camera, two Gaussians in front of it have depths closer than eps."""
    for camera in as_camera_list(cameras):
        z = scene.means @ camera.viewmat[:3, :3].T[:, 2] + camera.viewmat[2, 3]
        z = z[(z > near_plane) & (z < far_plane)].sort().values
        if z.numel() >= 2 and bool((z[1:] - z[:-1]).min() < eps):
            return True
    return False


def run_mr5(
    scene: GaussianScene,
    cameras: Union[Camera, Sequence[Camera]],
    seed: int,
    mode: Literal["correct", "misaligned"] = "correct",
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    tie_eps: float = 1e-6,
    perm: Optional[Tensor] = None,
) -> MR5Result:
    if mode not in ("correct", "misaligned"):
        raise ValueError(f"unknown mode: {mode}")

    # `perm` overrides the seeded derangement (e.g. `random_shuffle` to compare with earlier runs).
    if perm is None:
        perm = derangement(scene.n, seed)
    perm = perm.to(scene.means.device)
    colors_p = scene.colors if mode == "misaligned" else scene.colors[perm]

    tie = has_depth_tie(scene, cameras, near_plane, far_plane, tie_eps)
    viewmats, Ks, width, height = stack_cameras(cameras)

    common = dict(
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
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
