"""Pixel-level diagnosis of an MR1 (packed vs unpacked) difference on large-scale scenes.

Replays scene(s) as run_large_scale.py built them, renders with packed=True
and packed=False, and bisects (delta debugging, as in diagnose_mr4.py) a
smallest set of Gaussians that still reproduces the worst pixel's difference.
It prints the kept Gaussians' depths and alphas from each render's own gsplat
meta and ends with a `CLASS:` line (transmittance changed, depth order, or
unclassified) read off those values.

Usage:
    python gsplat_mt/diagnose_mr1.py --scene-idx 137
    python gsplat_mt/diagnose_mr1.py --scenes-from results/large_scale.csv --where mr1_verdict=isolated_discontinuity
"""

from __future__ import annotations

import argparse
import csv

import gsplat
import torch
from diagnose_mr4 import NEAR, FAR, bisect, load


def render_mode(cam, packed):
    def f(scene, with_meta):
        rgb, alpha, meta = gsplat.rasterization(
            means=scene.means,
            quats=scene.quats,
            scales=scene.scales,
            opacities=scene.opacities,
            colors=scene.colors,
            viewmats=cam.viewmat[None],
            Ks=cam.K[None],
            width=cam.width,
            height=cam.height,
            near_plane=NEAR,
            far_plane=FAR,
            sh_degree=None,
            packed=packed,
        )
        if with_meta:
            return rgb[0], alpha[0, ..., 0], meta
        return rgb[0], alpha[0, ..., 0]

    return f


def diagnose(a, idx):
    mode, scene, cameras = load(a, idx)
    cam = cameras[0]
    print(f"scene {idx} [{mode}] n={scene.n} res={cam.width}x{cam.height} cameras={len(cameras)}")
    # render_a = unpacked, render_b = packed; neither scales the scene (k_b = 1).
    return bisect(
        scene,
        cam,
        1.0,
        by=a.by,
        render_a=render_mode(cam, False),
        render_b=render_mode(cam, True),
        k_b=1.0,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-idx", type=int, default=None)
    p.add_argument("--scenes-from", type=str, default=None)
    p.add_argument("--where", type=str, default="mr1_verdict=isolated_discontinuity")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--by", choices=["color", "T"], default="color")
    a = p.parse_args()
    a.large, a.mode = True, None

    if a.scenes_from:
        col, val = a.where.split("=", 1)
        with open(a.scenes_from) as f:
            idxs = [int(r["scene_idx"]) for r in csv.DictReader(f) if r.get(col) == val]
    elif a.scene_idx is not None:
        idxs = [a.scene_idx]
    else:
        p.error("--scene-idx or --scenes-from is required")

    results = [(i, diagnose(a, i)) for i in idxs]
    print(f"\nSUMMARY of {len(results)} scenes:")
    counts = {}
    for i, r in results:
        c = r["class"] if r else "nothing over tolerance"
        counts[c] = counts.get(c, 0) + 1
        print(f"  scene {i}: {c}" + (f" (kept {r['n_keep']}, rel T change {r['rel_t']:.1e})" if r else ""))
    print(f"  counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
