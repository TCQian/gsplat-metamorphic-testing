"""Preliminary study: MR1-MR5 over small random scenes, a fixed number per scene type.

Every scene uses the same settings (64x64 images, 4 cameras, 20-200 Gaussians by default); only
the scene type varies. Writes one CSV row per scene (scene type, properties, D_max/D_MAE and a
verdict per relation), `<out>_summary.json` (per-type verdict counts) and `<out>_repro.json`
(standalone reproductions of gsplat issues #764 and #630). Requires a CUDA GPU.

Usage (from the repository root):
    python runners/run_preliminary.py --n-scenes 20 --out results/preliminary.csv
    python runners/run_preliminary.py --n-scenes 2 --modes default single --skip-repro --out results/smoke.csv
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # repository root, for `import gsplat_mt`

from gsplat_mt.harness import (
    build_argparser,
    count_verdicts,
    run_one_scene,
    summarize,
    sync,
)
from gsplat_mt.repro_issues import repro_issue_630, repro_issue_764
from gsplat_mt.scene import sample_cameras, sample_gaussians


def main(argv=None) -> int:
    args = build_argparser(__doc__).parse_args(argv)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print(
            "ERROR: CUDA is not available in this environment, but gsplat's "
            "rasterization kernels require an NVIDIA GPU (no CPU/MPS backend). "
            "Run this on a CUDA-capable machine. See README.md.",
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

    total = args.n_scenes * len(args.modes)
    print(
        f"Running {total} scenes ({args.n_scenes} per mode: {', '.join(args.modes)})..."
    )
    rows = []
    for m, mode in enumerate(args.modes):
        for j in range(args.n_scenes):
            i = m * args.n_scenes + j
            sync(device)
            t0 = time.perf_counter()
            try:
                row = run_one_scene(i, mode, args, device)
            except Exception as e:  # noqa: BLE001 - keep going, record the failure
                row = {
                    "scene_idx": i,
                    "seed": args.seed + i,
                    "mode": mode,
                    "fatal_error": f"{type(e).__name__}: {e}",
                }
            sync(device)
            row["time_s"] = time.perf_counter() - t0
            rows.append(row)
            print(f"  scene {i+1}/{total} [{mode}] done in {row['time_s']:.2f}s")

    fieldnames = sorted(set(itertools.chain.from_iterable(r.keys() for r in rows)))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")

    numeric_keys = [
        "mr1_rgb_d_max",
        "mr1_rgb_d_mae",
        "mr1_rgb_d_ssim",
        "mr2_max_d_max",
        "mr2_mean_d_mae",
        "mr2_mean_d_ssim",
        "mr3_correct_rgb_d_max",
        "mr3_correct_rgb_d_mae",
        "mr3_correct_rgb_d_ssim",
        "mr3_xyzwbug_rgb_d_max",
        "mr3_xyzwbug_rgb_d_mae",
        "mr3_xyzwbug_rgb_d_ssim",
        "mr4_rgb_d_max",
        "mr4_rgb_d_mae",
        "mr4_rgb_d_ssim",
        "mr4_exact_d_max",
        "mr4_bug_rgb_d_max",
        "mr5_rgb_d_max",
        "mr5_rgb_d_mae",
        "mr5_rgb_d_ssim",
        "mr5_bug_rgb_d_max",
        "time_s",
    ]
    summary = summarize(rows, numeric_keys)
    print("\nSummary (across scenes where the relation ran without error):")
    for key, stats in summary.items():
        print(
            f"  {key}: min={stats['min']:.3e} mean={stats['mean']:.3e} max={stats['max']:.3e} n={stats['n']}"
        )

    verdicts = count_verdicts(rows)
    print("\nVerdicts per mode (*xyzwbug / *_bug 'violation' = injected bug detected):")
    for mode, per_key in verdicts.items():
        print(f"  {mode}")
        for k, c in per_key.items():
            print(f"    {k}: {c}")

    n_errors = sum(
        1
        for r in rows
        if any(k.endswith("_error") and r.get(k) for k in r) or "fatal_error" in r
    )
    print(f"\nScenes with at least one MR error/crash: {n_errors}/{len(rows)}")

    with open(out_path.with_name(out_path.stem + "_summary.json"), "w") as f:
        json.dump({"stats": summary, "verdicts": verdicts}, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
