"""Synthetic Gaussian scene + pinhole camera generation.

Conventions (must match gsplat v1.5.3, see gsplat/rendering.py::rasterization):
  - quats are (w, x, y, z), not required to be pre-normalized by gsplat, but we
    normalize them ourselves so scale/quat and covariance parameterizations are
    built from the exact same rotation.
  - scales are raw (positive) values, not log-scale.
  - opacities are raw values in [0, 1], not passed through sigmoid.
  - viewmats are world-to-camera 4x4 (row-major, OpenCV-style: +X right, +Y down,
    +Z forward into the scene).
  - Ks are standard 3x3 pinhole intrinsics.

`sample_scene` adds edge-case modes (extreme opacity/scale, anisotropy, wide
spread, far-clipped, N=1) on top of the default `sample_gaussians` generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@dataclass
class GaussianScene:
    means: Tensor  # [N, 3]
    scales: Tensor  # [N, 3]
    quats: Tensor  # [N, 4], (w, x, y, z), normalized
    opacities: Tensor  # [N]
    colors: Tensor  # [N, 3]

    @property
    def n(self) -> int:
        return self.means.shape[0]

    def to(self, device) -> "GaussianScene":
        return GaussianScene(
            means=self.means.to(device),
            scales=self.scales.to(device),
            quats=self.quats.to(device),
            opacities=self.opacities.to(device),
            colors=self.colors.to(device),
        )


@dataclass
class Camera:
    viewmat: Tensor  # [4, 4], world-to-camera
    K: Tensor  # [3, 3]
    width: int
    height: int

    def to(self, device) -> "Camera":
        return Camera(
            viewmat=self.viewmat.to(device),
            K=self.K.to(device),
            width=self.width,
            height=self.height,
        )


def random_unit_quaternions(
    n: int,
    generator: Optional[torch.Generator] = None,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    """Uniform random rotations as (w, x, y, z) unit quaternions.

    Sampling a standard-normal 4-vector and normalizing it gives a uniform
    distribution on S^3 (the normal distribution is invariant to orthogonal
    transforms), which double-covers SO(3) uniformly.
    """
    q = torch.randn(n, 4, generator=generator, device=device, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return q


def sample_points_in_ball(
    n: int,
    radius: float,
    generator: Optional[torch.Generator] = None,
    device=None,
    dtype=torch.float32,
) -> Tensor:
    """Uniformly sample n points inside a 3D ball of the given radius."""
    directions = torch.randn(n, 3, generator=generator, device=device, dtype=dtype)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # cube root of a uniform variate gives a radius distribution uniform in volume.
    u = torch.rand(n, 1, generator=generator, device=device, dtype=dtype)
    r = radius * u.pow(1.0 / 3.0)
    return directions * r


def sample_gaussians(
    n: int,
    mean_radius: float = 0.6,
    scale_range: Tuple[float, float] = (0.02, 0.12),
    opacity_range: Tuple[float, float] = (0.2, 1.0),
    color_range: Tuple[float, float] = (0.0, 1.0),
    generator: Optional[torch.Generator] = None,
    device=None,
    dtype=torch.float32,
) -> GaussianScene:
    """Generate a random scene of n Gaussians centered near the world origin."""
    means = sample_points_in_ball(n, mean_radius, generator, device, dtype)
    quats = random_unit_quaternions(n, generator, device, dtype)

    lo, hi = scale_range
    scales = lo + (hi - lo) * torch.rand(
        n, 3, generator=generator, device=device, dtype=dtype
    )

    olo, ohi = opacity_range
    opacities = olo + (ohi - olo) * torch.rand(
        n, generator=generator, device=device, dtype=dtype
    )

    clo, chi = color_range
    colors = clo + (chi - clo) * torch.rand(
        n, 3, generator=generator, device=device, dtype=dtype
    )

    return GaussianScene(
        means=means, scales=scales, quats=quats, opacities=opacities, colors=colors
    )


SCENE_MODES = (
    "default",
    "opacity_extreme",
    "scale_tiny",
    "scale_huge",
    "anisotropic",
    "wide_spread",
    "far_clipped",
    "single",
)


def sample_scene(
    mode: str,
    n: int,
    generator: Optional[torch.Generator] = None,
    device=None,
    dtype=torch.float32,
    **kwargs,
) -> GaussianScene:
    """`sample_gaussians` with a stress mode applied.

    - default: unchanged.
    - opacity_extreme: opacities drawn from {0, 1e-4, 0.999, 1}.
    - scale_tiny / scale_huge: scales in [1e-4, 1e-3] / [0.15, 0.5].
    - anisotropic: one random axis per Gaussian shrunk 1000x (needles/discs).
    - wide_spread: means in a radius-8 ball, so many Gaussians fall behind the
      camera, outside the frustum, or inside the near plane.
    - far_clipped: radius-150 ball with large scales; most lie beyond far_plane.
    - single: N = 1.

    `kwargs` are the default-mode sampling arguments; a mode's own overrides win.
    """
    if mode not in SCENE_MODES:
        raise ValueError(f"unknown scene mode: {mode}")

    overrides = {
        "scale_tiny": {"scale_range": (1e-4, 1e-3)},
        "scale_huge": {"scale_range": (0.15, 0.5)},
        "wide_spread": {"mean_radius": 8.0},
        "far_clipped": {"mean_radius": 150.0, "scale_range": (1.0, 3.0)},
    }.get(mode, {})
    if mode == "single":
        n = 1

    scene = sample_gaussians(
        n, generator=generator, device=device, dtype=dtype, **{**kwargs, **overrides}
    )

    if mode == "opacity_extreme":
        choices = torch.tensor([0.0, 1e-4, 0.999, 1.0], dtype=dtype)
        idx = torch.randint(0, len(choices), (n,), generator=generator)
        scene.opacities = choices[idx].to(device=device)
    elif mode == "anisotropic":
        axis = torch.randint(0, 3, (n,), generator=generator)
        factor = torch.ones(n, 3, dtype=dtype)
        factor[torch.arange(n), axis] = 1e-3
        scene.scales = scene.scales * factor.to(device=device)

    return scene


def look_at_viewmat(eye: Tensor, target: Tensor, up: Tensor) -> Tensor:
    """World-to-camera 4x4 matrix for a camera at `eye` looking at `target`.

    Uses OpenCV camera axes: +X right, +Y down, +Z forward (into the scene).
    """
    forward = target - eye
    forward = forward / forward.norm().clamp_min(1e-12)

    right = torch.linalg.cross(forward, up)
    right_norm = right.norm()
    if right_norm < 1e-6:
        # forward is (near-)parallel to `up`; pick a different reference axis.
        alt_up = torch.tensor([1.0, 0.0, 0.0], dtype=eye.dtype, device=eye.device)
        right = torch.linalg.cross(forward, alt_up)
        right_norm = right.norm()
    right = right / right_norm.clamp_min(1e-12)

    true_down = torch.linalg.cross(forward, right)  # +Y down, OpenCV convention
    true_down = true_down / true_down.norm().clamp_min(1e-12)

    R = torch.stack([right, true_down, forward], dim=0)  # [3,3], world->cam rotation
    t = -R @ eye  # world->cam translation

    viewmat = torch.eye(4, dtype=eye.dtype, device=eye.device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    return viewmat


def pinhole_intrinsics(
    width: int, height: int, fov_deg: float, dtype=torch.float32, device=None
) -> Tensor:
    fov = math.radians(fov_deg)
    f = height / (2.0 * math.tan(fov / 2.0))
    K = torch.tensor(
        [
            [f, 0.0, width / 2.0],
            [0.0, f, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    return K


def sample_camera(
    width: int,
    height: int,
    dist_range: Tuple[float, float] = (2.5, 4.0),
    fov_deg: float = 60.0,
    azimuth_range: Tuple[float, float] = (0.0, 2 * math.pi),
    elevation_range: Tuple[float, float] = (-0.4, 0.4),
    generator: Optional[torch.Generator] = None,
    device=None,
    dtype=torch.float32,
) -> Camera:
    """Sample a pinhole camera on a sphere around the world origin.

    `dist_range` and `fov_deg` should be chosen so that a scene whose Gaussians
    lie within `mean_radius` of the origin comfortably fits inside the frustum,
    e.g. dist=2.5, fov=60deg gives a half-width of ~1.44 at the origin's depth
    plane, versus a scene radius of 0.6.
    """
    dlo, dhi = dist_range
    dist = dlo + (dhi - dlo) * torch.rand((), generator=generator).item()

    alo, ahi = azimuth_range
    azimuth = alo + (ahi - alo) * torch.rand((), generator=generator).item()

    elo, ehi = elevation_range
    elevation = elo + (ehi - elo) * torch.rand((), generator=generator).item()

    eye = torch.tensor(
        [
            dist * math.cos(elevation) * math.cos(azimuth),
            dist * math.sin(elevation),
            dist * math.cos(elevation) * math.sin(azimuth),
        ],
        dtype=dtype,
        device=device,
    )
    target = torch.zeros(3, dtype=dtype, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

    viewmat = look_at_viewmat(eye, target, up)
    K = pinhole_intrinsics(width, height, fov_deg, dtype=dtype, device=device)
    return Camera(viewmat=viewmat, K=K, width=width, height=height)


def sample_cameras(
    k: int,
    width: int,
    height: int,
    generator: Optional[torch.Generator] = None,
    device=None,
    dtype=torch.float32,
    **camera_kwargs,
) -> List[Camera]:
    return [
        sample_camera(
            width,
            height,
            generator=generator,
            device=device,
            dtype=dtype,
            **camera_kwargs,
        )
        for _ in range(k)
    ]
