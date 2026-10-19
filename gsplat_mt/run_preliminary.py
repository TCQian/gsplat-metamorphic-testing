"""Runs MR1-MR5 over random Gaussian scenes in each stress mode.

Writes one CSV row per scene (scene mode, properties, D_max/D_MAE and a
pass/violation/crash verdict per relation) plus a JSON summary of per-mode
verdict counts. Requires a CUDA GPU; see README.md.

Usage:
    python gsplat_mt/run_preliminary.py --n-scenes 20 --out results/refined_results.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import gsplat
import torch
from metrics import TOLERANCES, verdict
from mr1_packed import run_mr1
from mr2_batch import run_mr2
from mr3_covariance import run_mr3
from mr4_world_scale import run_mr4
from mr5_permutation import run_mr5
from repro_issues import repro_issue_630, repro_issue_764
from scene import SCENE_MODES, GaussianScene, sample_cameras, sample_gaussians, sample_scene


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


def render_stats(scene: GaussianScene, cameras, near_plane=0.01, far_plane=100.0) -> Dict[str, object]:
    """Visibility of the scene across `cameras`, to flag renders that draw nothing.

    Coverage comes from the rendered alpha. Visible-Gaussian counts come from
    gsplat's own culling (`meta["radii"] > 0`) and are None if that is unavailable.
    """
    try:
        _, alpha, meta = gsplat.rasterization(
            means=scene.means,
            quats=scene.quats,
            scales=scene.scales,
            opacities=scene.opacities,
            colors=scene.colors,
            viewmats=torch.stack([c.viewmat for c in cameras]),
            Ks=torch.stack([c.K for c in cameras]),
            width=cameras[0].width,
            height=cameras[0].height,
            near_plane=near_plane,
            far_plane=far_plane,
            sh_degree=None,
            packed=False,
        )
    except Exception as e:  # noqa: BLE001 - the MRs will surface the same crash
        return {"row": {"render_stats_error": f"{type(e).__name__}: {e}"}}

    coverage = (alpha > 0).float().flatten(1).mean(dim=1)  # [C]
    row: Dict[str, object] = {
        "coverage_cam0": coverage[0].item(),
        "coverage_mean": coverage.mean().item(),
        "n_empty_cams": int((coverage == 0).sum().item()),
        "n_visible_cam0": None,
        "n_visible_any": None,
    }
    try:
        radii = meta["radii"]
        visible = (radii > 0).all(dim=-1) if radii.dim() == 3 else radii > 0  # [C,N]
        row["n_visible_cam0"] = int(visible[0].sum().item())
        row["n_visible_any"] = int(visible.any(dim=0).sum().item())
    except Exception:  # noqa: BLE001 - radii layout differs; coverage still valid
        pass
    return {"row": row, "cam0_empty": coverage[0].item() == 0, "all_empty": bool((coverage == 0).all())}


def flag_vacuous(row: Dict[str, object], keys, empty: bool) -> None:
    """A 'pass' on a render that draws nothing proves nothing."""
    if empty:
        for k in keys:
            if row.get(k) == "pass":
                row[k] = "vacuous_pass"


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


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
    mode: str,
    args: argparse.Namespace,
    device: str,
) -> Dict[str, object]:
    seed = args.seed + scene_idx
    generator = torch.Generator(device="cpu").manual_seed(seed)

    n_gaussians = torch.randint(
        args.n_gaussians_min, args.n_gaussians_max + 1, (1,), generator=generator
    ).item()

    scene = sample_scene(
        mode,
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

    row: Dict[str, object] = {"scene_idx": scene_idx, "seed": seed, "mode": mode}
    row.update(scene_properties(scene, cameras))
    stats = render_stats(scene, cameras)
    row.update(stats["row"])

    mr1 = run_mr1(scene, cameras[0])
    row.update(flatten_diff("mr1_rgb", mr1.rgb, mr1.error))
    row.update(flatten_diff("mr1_alpha", mr1.alpha, mr1.error))
    row["mr1_verdict"] = verdict((mr1.rgb, mr1.alpha), mr1.error, TOLERANCES["mr1"])

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
    row["mr2_verdict"] = verdict(tuple(mr2.per_camera), mr2.error, TOLERANCES["mr2"])

    mr3_correct = run_mr3(scene, cameras[0], mode="correct")
    row.update(flatten_diff("mr3_correct_rgb", mr3_correct.rgb, mr3_correct.error))
    row["mr3_correct_verdict"] = verdict(
        (mr3_correct.rgb, mr3_correct.alpha), mr3_correct.error, TOLERANCES["mr3"]
    )

    mr3_bug = run_mr3(scene, cameras[0], mode="xyzw_bug")
    row.update(flatten_diff("mr3_xyzwbug_rgb", mr3_bug.rgb, mr3_bug.error))
    # Fault injection: "violation" here means the injected bug was detected.
    row["mr3_xyzwbug_verdict"] = verdict(
        (mr3_bug.rgb, mr3_bug.alpha), mr3_bug.error, TOLERANCES["mr3"]
    )

    # MR4: sweep world-scale factors; record the worst one.
    mr4_runs = [(k, run_mr4(scene, cameras[0], k)) for k in args.world_scales]
    mr4_diffs = tuple(d for _, r in mr4_runs for d in (r.rgb, r.alpha))
    mr4_error = next((r.error for _, r in mr4_runs if r.error), None)
    ok = [(k, r) for k, r in mr4_runs if r.rgb is not None]
    worst_k, worst = max(ok, key=lambda kr: kr[1].rgb.d_max) if ok else (None, None)
    row.update(flatten_diff("mr4_rgb", worst.rgb if worst else None, mr4_error))
    row["mr4_worst_k"] = worst_k
    row["mr4_verdict"] = verdict(mr4_diffs, mr4_error, TOLERANCES["mr4"])

    # Fault injection: camera translation not scaled, at the largest factor.
    mr4_bug = run_mr4(scene, cameras[0], max(args.world_scales), mode="no_translation")
    row.update(flatten_diff("mr4_bug_rgb", mr4_bug.rgb, mr4_bug.error))
    row["mr4_bug_verdict"] = verdict(
        (mr4_bug.rgb, mr4_bug.alpha), mr4_bug.error, TOLERANCES["mr4"]
    )

    mr5 = run_mr5(scene, cameras[0], seed)
    row.update(flatten_diff("mr5_rgb", mr5.rgb, mr5.error))
    row["mr5_depth_tie"] = mr5.depth_tie
    row["mr5_verdict"] = verdict((mr5.rgb, mr5.alpha), mr5.error, TOLERANCES["mr5"])
    if mr5.depth_tie and row["mr5_verdict"] == "violation":
        row["mr5_verdict"] = "tie_ambiguous"  # compositing order of tied depths is ambiguous

    # Fault injection: colors not permuted along with the other attributes.
    mr5_bug = run_mr5(scene, cameras[0], seed, mode="misaligned")
    row.update(flatten_diff("mr5_bug_rgb", mr5_bug.rgb, mr5_bug.error))
    row["mr5_bug_verdict"] = verdict(
        (mr5_bug.rgb, mr5_bug.alpha), mr5_bug.error, TOLERANCES["mr5"]
    )

    # All relations except MR2 render cameras[0] only; MR2 uses every camera.
    flag_vacuous(row, [k for k in VERDICT_KEYS if k != "mr2_verdict"], stats.get("cam0_empty", False))
    flag_vacuous(row, ["mr2_verdict"], stats.get("all_empty", False))

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


VERDICT_KEYS = [
    "mr1_verdict",
    "mr2_verdict",
    "mr3_correct_verdict",
    "mr3_xyzwbug_verdict",
    "mr4_verdict",
    "mr4_bug_verdict",
    "mr5_verdict",
    "mr5_bug_verdict",
]


def count_verdicts(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """{mode: {verdict_key: {verdict: count}}}; scenes that crashed outright count as 'fatal'."""
    counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    for r in rows:
        per_mode = counts.setdefault(r["mode"], {k: {} for k in VERDICT_KEYS})
        for k in VERDICT_KEYS:
            v = r.get(k, "fatal")
            per_mode[k][v] = per_mode[k].get(v, 0) + 1
    return counts


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-scenes", type=int, default=20, help="scenes per mode")
    p.add_argument(
        "--modes",
        nargs="+",
        default=list(SCENE_MODES),
        choices=SCENE_MODES,
        help="scene generator modes to run",
    )
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
    p.add_argument(
        "--world-scales",
        type=float,
        nargs="+",
        default=[0.01, 0.25, 4.0, 100.0],
        help="MR4 world-scale factors k",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="results/refined_results.csv")
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
    print(f"Running {total} scenes ({args.n_scenes} per mode: {', '.join(args.modes)})...")
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
        "mr2_max_d_max",
        "mr2_mean_d_mae",
        "mr3_correct_rgb_d_max",
        "mr3_correct_rgb_d_mae",
        "mr3_xyzwbug_rgb_d_max",
        "mr3_xyzwbug_rgb_d_mae",
        "mr4_rgb_d_max",
        "mr4_rgb_d_mae",
        "mr4_bug_rgb_d_max",
        "mr5_rgb_d_max",
        "mr5_rgb_d_mae",
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
