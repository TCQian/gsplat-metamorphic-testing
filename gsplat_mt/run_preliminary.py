"""Preliminary-study runner (main.tex, Section "Preliminary Study").

Generates a small set of random Gaussian scenes, runs MR1/MR2/MR3 on each,
and writes one CSV row per scene with the recorded scene properties (number
of Gaussians, scale, opacity, depth, overlap, number of cameras) and the
D_max/D_MAE of each metamorphic relation, plus a text summary.

Must be run on a CUDA-capable machine -- see requirements.txt / README.md.

Usage:
    python -m gsplat_mt.run_preliminary --n-scenes 20 --out results/preliminary.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from mr1_packed import run_mr1
from mr2_batch import run_mr2
from mr3_covariance import run_mr3
from repro_issues import repro_issue_630, repro_issue_764
from scene import GaussianScene, sample_cameras, sample_gaussians


def scene_properties(scene: GaussianScene, cameras) -> Dict[str, float]:
    n = scene.n
    means = scene.means
    scales = scene.scales
    opacities = scene.opacities

    depths = []
    for cam in cameras:
        R = cam.viewmat[:3, :3]
        t = cam.viewmat[:3, 3]
        cam_pts = means @ R.T + t
        depths.append(cam_pts[:, 2].mean().item())

    # Overlap proxy: fraction of Gaussian pairs whose mean-to-mean distance is
    # smaller than the sum of their largest-axis scales (a coarse anisotropic
    # bounding-sphere overlap test, cheap enough for the small N used here).
    overlap_frac = float("nan")
    if n >= 2:
        dists = torch.cdist(means, means)  # [N,N]
        max_scale = scales.max(dim=-1).values  # [N]
        size_sum = max_scale[:, None] + max_scale[None, :]  # [N,N]
        iu = torch.triu_indices(n, n, offset=1)
        pair_dists = dists[iu[0], iu[1]]
        pair_sizes = size_sum[iu[0], iu[1]]
        overlap_frac = (pair_dists < pair_sizes).float().mean().item()

    return {
        "n_gaussians": n,
        "n_cameras": len(cameras),
        "mean_scale": scales.mean().item(),
        "max_scale": scales.max().item(),
        "mean_opacity": opacities.mean().item(),
        "mean_depth": sum(depths) / len(depths),
        "min_depth": min(depths),
        "overlap_frac": overlap_frac,
    }


def flatten_diff(prefix: str, diff, error: str = None) -> Dict[str, object]:
    if diff is None:
        return {
            f"{prefix}_d_max": None,
            f"{prefix}_d_mae": None,
            f"{prefix}_has_nan": None,
            f"{prefix}_has_inf": None,
            f"{prefix}_error": error,
        }
    row = diff.as_row(prefix)
    row[f"{prefix}_error"] = error
    return row


def run_one_scene(
    scene_idx: int,
    args: argparse.Namespace,
    device: str,
) -> Dict[str, object]:
    seed = args.seed + scene_idx
    generator = torch.Generator(device="cpu").manual_seed(seed)

    n_gaussians = torch.randint(
        args.n_gaussians_min, args.n_gaussians_max + 1, (1,), generator=generator
    ).item()

    scene = sample_gaussians(
        n_gaussians,
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

    row: Dict[str, object] = {"scene_idx": scene_idx, "seed": seed}
    row.update(scene_properties(scene, cameras))

    mr1 = run_mr1(scene, cameras[0])
    row.update(flatten_diff("mr1_rgb", mr1.rgb, mr1.error))
    row.update(flatten_diff("mr1_alpha", mr1.alpha, mr1.error))

    if args.mr1_backgrounds:
        mr1_bg = run_mr1(scene, cameras[0], use_backgrounds=True)
        row.update(flatten_diff("mr1_bg_rgb", mr1_bg.rgb, mr1_bg.error))

    mr2 = run_mr2(scene, cameras)
    if mr2.error is not None:
        row["mr2_max_d_max"] = None
        row["mr2_mean_d_mae"] = None
        row["mr2_error"] = mr2.error
    else:
        row["mr2_max_d_max"] = max(d.d_max for d in mr2.per_camera)
        row["mr2_mean_d_mae"] = sum(d.d_mae for d in mr2.per_camera) / len(
            mr2.per_camera
        )
        row["mr2_error"] = None

    mr3_correct = run_mr3(scene, cameras[0], mode="correct")
    row.update(flatten_diff("mr3_correct_rgb", mr3_correct.rgb, mr3_correct.error))

    mr3_bug = run_mr3(scene, cameras[0], mode="xyzw_bug")
    row.update(flatten_diff("mr3_xyzwbug_rgb", mr3_bug.rgb, mr3_bug.error))

    return row


def summarize(
    rows: List[Dict[str, object]], keys: List[str]
) -> Dict[str, Dict[str, float]]:
    summary = {}
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        summary[key] = {
            "min": min(vals),
            "mean": sum(vals) / len(vals),
            "max": max(vals),
            "n": len(vals),
        }
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-scenes", type=int, default=20)
    p.add_argument("--n-gaussians-min", type=int, default=20)
    p.add_argument("--n-gaussians-max", type=int, default=200)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--n-cameras", type=int, default=4)
    p.add_argument("--mean-radius", type=float, default=0.6)
    p.add_argument("--scale-min", type=float, default=0.02)
    p.add_argument("--scale-max", type=float, default=0.12)
    p.add_argument("--opacity-min", type=float, default=0.2)
    p.add_argument("--opacity-max", type=float, default=1.0)
    p.add_argument("--dist-min", type=float, default=2.5)
    p.add_argument("--dist-max", type=float, default=4.0)
    p.add_argument("--fov-deg", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="results/preliminary_results.csv")
    p.add_argument(
        "--mr1-backgrounds",
        action="store_true",
        help="also run MR1 with a random background color, to probe gsplat issue #764",
    )
    p.add_argument(
        "--skip-repro",
        action="store_true",
        help="skip the issue #764/#630 repro checks",
    )
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print(
            "ERROR: CUDA is not available in this environment, but gsplat's "
            "rasterization kernels require an NVIDIA GPU (no CPU/MPS backend). "
            "Run this on a CUDA-capable machine, e.g. a Google Colab GPU runtime. "
            "See README.md.",
            file=sys.stderr,
        )
        return 1

    device = args.device
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_repro:
        print("Running standalone issue reproductions...")
        r764 = repro_issue_764(device=device)
        print(f"  {r764.name}: detected={r764.detected} :: {r764.detail}")

        gen = torch.Generator(device="cpu").manual_seed(args.seed - 1)
        probe_scene = sample_gaussians(50, generator=gen).to(device)
        probe_cam = sample_cameras(1, args.width, args.height, generator=gen)[0].to(
            device
        )
        r630 = repro_issue_630(probe_scene, probe_cam)
        print(f"  {r630.name}: detected={r630.detected} :: {r630.detail}")

        with open(out_path.with_name(out_path.stem + "_repro.json"), "w") as f:
            json.dump(
                {
                    r764.name: {"detected": r764.detected, "detail": r764.detail},
                    r630.name: {"detected": r630.detected, "detail": r630.detail},
                },
                f,
                indent=2,
            )

    print(f"Running {args.n_scenes} scenes...")
    rows = []
    for i in range(args.n_scenes):
        try:
            row = run_one_scene(i, args, device)
        except Exception as e:  # noqa: BLE001 - keep going, record the failure
            row = {
                "scene_idx": i,
                "seed": args.seed + i,
                "fatal_error": f"{type(e).__name__}: {e}",
            }
        rows.append(row)
        print(f"  scene {i+1}/{args.n_scenes} done")

    fieldnames = sorted(set(itertools.chain.from_iterable(r.keys() for r in rows)))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")

    numeric_keys = [
        "mr1_rgb_d_max",
        "mr1_rgb_d_mae",
        "mr2_max_d_max",
        "mr2_mean_d_mae",
        "mr3_correct_rgb_d_max",
        "mr3_correct_rgb_d_mae",
        "mr3_xyzwbug_rgb_d_max",
        "mr3_xyzwbug_rgb_d_mae",
    ]
    summary = summarize(rows, numeric_keys)
    print("\nSummary (across scenes where the relation ran without error):")
    for key, stats in summary.items():
        print(
            f"  {key}: min={stats['min']:.3e} mean={stats['mean']:.3e} max={stats['max']:.3e} n={stats['n']}"
        )

    n_errors = sum(
        1
        for r in rows
        if any(k.endswith("_error") and r.get(k) for k in r) or "fatal_error" in r
    )
    print(f"\nScenes with at least one MR error/crash: {n_errors}/{len(rows)}")

    with open(out_path.with_name(out_path.stem + "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
