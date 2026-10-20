"""Pixel-level diagnosis of an MR4 (world-scale) difference on one scene.

Rebuilds scene `--scene-idx` exactly as run_preliminary.py did, then for each
factor k renders the original and the k-scaled scene and reports which pixels
differ. Power-of-two k scale exactly in binary, so a difference that vanishes
there is rounding; one that persists is not.

Usage:
    python gsplat_mt/diagnose_mr4.py --scene-idx 68 --mode scale_huge
"""

from __future__ import annotations

import argparse

import gsplat
import torch
from run_preliminary import build_argparser
from scene import sample_cameras, sample_scene

NEAR, FAR = 0.01, 100.0


def build_scene(scene_idx, mode, args, device):
    """Mirrors the scene/camera construction in run_preliminary.run_one_scene."""
    generator = torch.Generator(device="cpu").manual_seed(args.seed + scene_idx)
    n = torch.randint(
        args.n_gaussians_min, args.n_gaussians_max + 1, (1,), generator=generator
    ).item()
    scene = sample_scene(
        mode,
        n,
        mean_radius=args.mean_radius,
        scale_range=(args.scale_min, args.scale_max),
        opacity_range=(args.opacity_min, args.opacity_max),
        generator=generator,
    ).to(device)
    cameras = [
        c.to(device)
        for c in sample_cameras(
            args.n_cameras,
            args.width,
            args.height,
            dist_range=(args.dist_min, args.dist_max),
            fov_deg=args.fov_deg,
            generator=generator,
        )
    ]
    return scene, cameras


def render(scene, cam, k):
    viewmat = cam.viewmat.clone()
    viewmat[:3, 3] *= k
    rgb, alpha, _ = gsplat.rasterization(
        means=scene.means * k,
        quats=scene.quats,
        scales=scene.scales * k,
        opacities=scene.opacities,
        colors=scene.colors,
        viewmats=viewmat[None],
        Ks=cam.K[None],
        width=cam.width,
        height=cam.height,
        near_plane=NEAR * k,
        far_plane=FAR * k,
        sh_degree=None,
        packed=False,
    )
    return rgb[0], alpha[0, ..., 0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-idx", type=int, required=True)
    p.add_argument("--mode", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ks", type=float, nargs="+", default=[1.0, 0.01, 0.0078125, 0.25, 4.0, 100.0, 128.0])
    p.add_argument("--top", type=int, default=5)
    a = p.parse_args()

    args = build_argparser().parse_args([])
    args.seed = a.seed
    scene, cameras = build_scene(a.scene_idx, a.mode, args, "cuda")
    cam = cameras[0]
    print(f"scene {a.scene_idx} [{a.mode}] n={scene.n}")

    rgb0, alpha0 = render(scene, cam, 1.0)
    rgb0b, _ = render(scene, cam, 1.0)
    print(f"repeat render (determinism): D_max={(rgb0 - rgb0b).abs().max().item():.3e}")

    for k in a.ks:
        rgb, alpha = render(scene, cam, k)
        diff = (rgb - rgb0).abs().amax(-1)  # [H,W]
        pow2 = float(torch.log2(torch.tensor(k)).item()).is_integer()
        print(
            f"\nk={k:g} ({'power of 2' if pow2 else 'not exact'}): "
            f"D_max={diff.max().item():.3e}  D_mae={(rgb - rgb0).abs().mean().item():.3e}  "
            f"pixels differing: >0: {(diff > 0).sum().item()}, >1e-6: {(diff > 1e-6).sum().item()}, "
            f">1e-4: {(diff > 1e-4).sum().item()}"
        )
        w = diff.shape[1]
        vals, idx = diff.flatten().topk(a.top)
        for v, i in zip(vals.tolist(), idx.tolist()):
            y, x = divmod(i, w)
            if v == 0:
                break
            print(
                f"  (y={y:2d}, x={x:2d}) diff={v:.3e}  alpha: base={alpha0[y, x].item():.6f} "
                f"scaled={alpha[y, x].item():.6f}  1-alpha base={1 - alpha0[y, x].item():.3e} "
                f"scaled={1 - alpha[y, x].item():.3e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
