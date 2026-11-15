"""Pixel-level diagnosis of an MR1 (packed vs unpacked) difference on large-scale scenes.

Replays scene(s) as runners/run_large_scale.py built them (by default the
all-camera run; `--seed 1000 --modes` with the original eight types for the
camera-0 runs), renders with packed=True and packed=False on the camera that
shows the difference (`--camera`, as in diagnose_mr4.py), and bisects (delta
debugging) a smallest set of Gaussians that still reproduces the worst pixel's
difference.
It prints the kept Gaussians' depths and alphas from each render's own gsplat
meta and ends with a `CLASS:` line (transmittance changed, depth order, or
unclassified) read off those values.

Usage (from the repository root):
    python diagnosis/diagnose_mr1.py --scene-idx 42081
    python diagnosis/diagnose_mr1.py --scenes-from results/allcam_100000_findings.csv --where mr1_verdict=isolated_discontinuity --every 10
"""

from __future__ import annotations

import sys
from pathlib import Path

import argparse
import csv

import gsplat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, for `import gsplat_mt`

from diagnosis.diagnose_mr4 import FAR, NEAR, add_camera_arg, bisect, choose_camera, load
from gsplat_mt.scene import SCENE_MODES


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
    c = choose_camera(
        a.camera, cameras, lambda cm: render_mode(cm, False)(scene, False), lambda cm: render_mode(cm, True)(scene, False)
    )
    cam = cameras[c]
    print(f"scene {idx} [{mode}] n={scene.n} res={cam.width}x{cam.height} cameras={len(cameras)} camera={c}")
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
    p.add_argument("--every", type=int, default=1, help="with --scenes-from, bisect every K-th matching scene")
    p.add_argument("--offset", type=int, default=0, help="with --every, start at the OFFSET-th matching scene")
    p.add_argument("--seed", type=int, default=None, help="seed of the run being replayed (default 2000, the all-camera run)")
    p.add_argument("--modes", nargs="+", default=None, choices=SCENE_MODES, help="scene types of the run (default all nine)")
    p.add_argument("--by", choices=["color", "T"], default="color")
    add_camera_arg(p)
    a = p.parse_args()
    a.large, a.mode = True, None

    if a.scenes_from:
        col, val = a.where.split("=", 1)
        with open(a.scenes_from) as f:
            idxs = [int(r["scene_idx"]) for r in csv.DictReader(f) if r.get(col) == val][a.offset :: a.every]
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
