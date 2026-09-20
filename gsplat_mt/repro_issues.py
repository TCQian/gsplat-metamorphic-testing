"""Reproductions of previously reported gsplat issues referenced in the
proposal (main.tex, Motivation / Preliminary Study), used to sanity-check that
the installed gsplat==1.5.3 behaves as expected before trusting MR results.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from metrics import ImageDiff, compare_images
from mr3_covariance import _covars_xyzw_bug


@dataclass
class ReproResult:
    name: str
    detected: bool
    detail: str


def repro_issue_764(device: str = "cuda") -> ReproResult:
    """https://github.com/nerfstudio-project/gsplat/issues/764

    In gsplat 1.5.3, `packed=True` combined with `backgrounds` raises an
    AssertionError in `rasterize_to_pixels()`. Minimal repro adapted from the
    issue. Returns detected=True if the AssertionError is reproduced.
    """
    import gsplat

    means = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[1.0, 1.0, 1.0]], device=device)
    opacities = torch.tensor([0.8], device=device)
    colors = torch.tensor([[0.8, 0.2, 0.2]], device=device)

    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 4.0
    width, height = 512, 512
    K = torch.tensor(
        [[width, 1.0, width / 2], [1.0, height, height / 2], [0.0, 0.0, 1.0]],
        device=device,
    )
    background = torch.tensor([0.7, 0.9, 0.9], device=device)

    try:
        gsplat.rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmat.unsqueeze(0),
            K.unsqueeze(0),
            width,
            height,
            backgrounds=background.unsqueeze(0),
            packed=True,
        )
    except AssertionError as e:
        return ReproResult("issue_764_packed_backgrounds_assert", True, str(e))
    except Exception as e:  # noqa: BLE001
        return ReproResult(
            "issue_764_packed_backgrounds_assert",
            False,
            f"expected AssertionError, got {type(e).__name__}: {e}",
        )
    return ReproResult(
        "issue_764_packed_backgrounds_assert",
        False,
        "no exception raised; issue may already be fixed in this gsplat build",
    )


def repro_issue_630(
    scene, camera, near_plane: float = 0.01, far_plane: float = 100.0
) -> ReproResult:
    """https://github.com/nerfstudio-project/gsplat/issues/630

    Compares gsplat's own (correct, wxyz) scale/quat-to-covariance conversion
    against the reporter's xyzw-unbind formula applied to the same quats. See
    mr3_covariance.py for why this reproduces the reported discrepancy as a
    quaternion-convention bug in the *caller's* code rather than in gsplat.
    """
    import gsplat

    covars_correct, _ = gsplat.quat_scale_to_covar_preci(
        scene.quats, scene.scales, compute_covar=True, compute_preci=False, triu=False
    )
    covars_bug = _covars_xyzw_bug(scene.quats, scene.scales)

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
        packed=False,
    )

    rgb_sq, _, _ = gsplat.rasterization(covars=None, **common)
    rgb_correct, _, _ = gsplat.rasterization(covars=covars_correct, **common)
    rgb_bug, _, _ = gsplat.rasterization(covars=covars_bug, **common)

    diff_correct: ImageDiff = compare_images(rgb_sq, rgb_correct)
    diff_bug: ImageDiff = compare_images(rgb_sq, rgb_bug)

    detected = diff_bug.d_max > 10 * max(diff_correct.d_max, 1e-8)
    detail = (
        f"correct-covar D_max={diff_correct.d_max:.3e} D_mae={diff_correct.d_mae:.3e} | "
        f"xyzw-bug-covar D_max={diff_bug.d_max:.3e} D_mae={diff_bug.d_mae:.3e}"
    )
    return ReproResult("issue_630_covar_quat_convention", detected, detail)
