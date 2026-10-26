"""Large-scale randomized MR1-MR5 testing.

Every scene draws its own configuration (number of Gaussians, scene radius,
scale and opacity ranges, camera distance, field of view, number of cameras,
resolution) plus a stress mode, then runs the same per-scene pipeline as
run_preliminary.py. A mode's own overrides win over the sampled values it
touches (e.g. wide_spread fixes mean_radius), so cfg_* columns do not apply to
those fields in that mode.

Writes <out>.csv (all scenes), <out>.jsonl (one row per scene, flushed as it
goes so a timeout keeps partial results), <out>_findings.csv (scenes with a
crash, violation, isolated_discontinuity, tie_ambiguous or invalid_output in
MR1-MR5) and <out>_summary.json (stats, verdict counts per mode, fault-injection
detection rates, Spearman correlations of scene properties with MR1/MR4 D_max).

Usage:
    python gsplat_mt/run_large_scale.py --n-scenes 2000 --out results/large_scale.csv
    python gsplat_mt/run_large_scale.py --only-scene 137 --out results/one.csv   # replay one scene
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from run_preliminary import (
    build_argparser,
    count_verdicts,
    run_one_scene,
    summarize,
    sync,
)
from scene import SCENE_MODES

RESOLUTIONS = [(64, 64), (120, 90), (128, 128), (256, 192), (256, 256)]
MR_KEYS = ["mr1_verdict", "mr2_verdict", "mr3_correct_verdict", "mr4_verdict", "mr4_exact_verdict", "mr5_verdict"]
BUG_KEYS = ["mr3_xyzwbug_verdict", "mr4_bug_verdict", "mr5_bug_verdict"]
UNREMARKABLE = ("pass", "vacuous_pass")
CORR_PROPS = [
    "n_gaussians",
    "n_cameras",
    "mean_scale",
    "mean_opacity",
    "mean_depth",
    "overlap_frac",
    "coverage_cam0",
    "opaque_frac_cam0",
    "cfg_pixels",
]
CORR_TARGETS = ["mr1_rgb_d_max", "mr4_rgb_d_max"]


def loguniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def sample_config(scene_idx: int, args: argparse.Namespace) -> Dict[str, object]:
    """Deterministic per-scene configuration (independent of the scene's own seed)."""
    rng = random.Random(f"{args.seed}:{scene_idx}")
    n = int(round(loguniform(rng, args.n_min, args.n_max)))
    width, height = rng.choice(RESOLUTIONS)
    radius = loguniform(rng, 0.3, 2.0)
    scale_c = loguniform(rng, 0.01, 0.3)
    op_lo = rng.uniform(0.05, 0.9)
    dist_min = radius * rng.uniform(1.2, 8.0)
    return {
        "mode": rng.choice(args.modes),
        "n": n,
        "width": width,
        "height": height,
        "n_cameras": rng.randint(1, args.max_cameras),
        "mean_radius": radius,
        "scale_min": scale_c * 0.5,
        "scale_max": scale_c * 1.5,
        "opacity_min": op_lo,
        "opacity_max": rng.uniform(op_lo, 1.0),
        "dist_min": dist_min,
        "dist_max": dist_min * rng.uniform(1.0, 1.6),
        "fov_deg": rng.uniform(30.0, 90.0),
    }


def scene_args(cfg: Dict[str, object], args: argparse.Namespace) -> argparse.Namespace:
    """Namespace with the fields run_one_scene reads, set from `cfg`."""
    ns = build_argparser().parse_args([])
    ns.seed = args.seed
    ns.world_scales = args.world_scales
    ns.n_gaussians_min = ns.n_gaussians_max = cfg["n"]
    for k in (
        "width",
        "height",
        "n_cameras",
        "mean_radius",
        "scale_min",
        "scale_max",
        "opacity_min",
        "opacity_max",
        "dist_min",
        "dist_max",
        "fov_deg",
    ):
        setattr(ns, k, cfg[k])
    return ns


def _ranks(v: torch.Tensor) -> torch.Tensor:
    _, inv, cnt = torch.unique(v, return_inverse=True, return_counts=True)
    ends = cnt.cumsum(0).double()
    return ((ends - cnt + 1 + ends) / 2)[inv]


def spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 10:
        return None
    rx = _ranks(torch.tensor(x, dtype=torch.float64))
    ry = _ranks(torch.tensor(y, dtype=torch.float64))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return torch.corrcoef(torch.stack([rx, ry]))[0, 1].item()


def correlations(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for t in CORR_TARGETS:
        out[t] = {}
        for p in CORR_PROPS:
            pairs = [
                (float(r[p]), float(r[t]))
                for r in rows
                if isinstance(r.get(p), (int, float))
                and isinstance(r.get(t), (int, float))
                and math.isfinite(r[p])
                and math.isfinite(r[t])
            ]
            out[t][p] = spearman([a for a, _ in pairs], [b for _, b in pairs])
    return out


def detection_rates(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    """Fault-injection outcomes (violation = injected bug detected) over all scenes."""
    rates: Dict[str, Dict[str, int]] = {}
    for k in BUG_KEYS:
        rates[k] = {}
        for r in rows:
            v = r.get(k, "fatal")
            rates[k][v] = rates[k].get(v, 0) + 1
    return rates


def timing_summary(rows: List[Dict[str, object]], wall_s: float) -> Dict[str, object]:
    """Wall-clock plus per-scene time. Scene 0 is reported separately: it includes the one-time CUDA JIT compile."""
    rest = [r for r in rows[1:] if r.get("time_s") is not None]

    def pct(v: List[float], q: float) -> float:
        return sorted(v)[int(q * (len(v) - 1))]

    out: Dict[str, object] = {
        "total_wall_s": wall_s,
        "first_scene_s": rows[0]["time_s"] if rows else None,
    }
    if rest:
        t = [r["time_s"] for r in rest]
        out.update(
            rest_sum_s=sum(t),
            rest_median_s=pct(t, 0.5),
            rest_p95_s=pct(t, 0.95),
            rest_p99_s=pct(t, 0.99),
            rest_max_s=max(t),
            per_mode_median_s={
                m: pct([r["time_s"] for r in rest if r["mode"] == m], 0.5)
                for m in sorted({r["mode"] for r in rest})
            },
        )
    return out


def is_finding(row: Dict[str, object]) -> bool:
    return "fatal_error" in row or any(
        row.get(k) not in UNREMARKABLE for k in MR_KEYS if k in row
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-scenes", type=int, default=2000)
    p.add_argument("--modes", nargs="+", default=list(SCENE_MODES), choices=SCENE_MODES)
    p.add_argument("--n-min", type=int, default=10)
    p.add_argument("--n-max", type=int, default=5000)
    p.add_argument("--max-cameras", type=int, default=8)
    p.add_argument("--world-scales", type=float, nargs="+", default=[0.01, 0.0078125, 0.25, 4.0, 100.0, 128.0])
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--only-scene", type=int, default=None, help="run just this scene index")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="results/large_scale.csv")
    args = p.parse_args(argv)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print(
            "ERROR: CUDA is not available in this environment. "
            "Run this on a CUDA-capable machine. See README.md.",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = out.with_suffix(".jsonl")

    indices = [args.only_scene] if args.only_scene is not None else range(args.n_scenes)
    total = len(indices)
    print(f"Running {total} randomized scenes...")
    rows: List[Dict[str, object]] = []
    aborted = None
    run_start = time.perf_counter()
    with open(jsonl_path, "w") as jf:
        for n_done, i in enumerate(indices, 1):
            cfg = sample_config(i, args)
            sync(args.device)
            t0 = time.perf_counter()
            try:
                row = run_one_scene(i, cfg["mode"], scene_args(cfg, args), args.device)
            except Exception as e:  # noqa: BLE001 - keep going, record the failure
                row = {
                    "scene_idx": i,
                    "seed": args.seed + i,
                    "mode": cfg["mode"],
                    "fatal_error": f"{type(e).__name__}: {e}",
                }
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                elif "CUDA" in str(e):
                    aborted = f"scene {i}: {row['fatal_error']}"  # sticky CUDA error poisons every later scene
            if aborted is None:
                sync(args.device)
            row["time_s"] = time.perf_counter() - t0
            row.update({f"cfg_{k}": v for k, v in cfg.items() if k != "mode"})
            row["cfg_pixels"] = cfg["width"] * cfg["height"]
            rows.append(row)
            jf.write(json.dumps(row) + "\n")
            jf.flush()
            if n_done == 1 or n_done % 100 == 0 or n_done == total:
                print(
                    f"  {n_done}/{total} done "
                    f"({time.perf_counter() - run_start:.0f}s elapsed, last scene {row['time_s']:.2f}s)"
                )
            if aborted:
                print(f"ABORTING: CUDA error, later scenes would be invalid ({aborted})", file=sys.stderr)
                break

    fieldnames = sorted(set(itertools.chain.from_iterable(r.keys() for r in rows)))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")

    findings = [r for r in rows if is_finding(r)]
    with open(out.with_name(out.stem + "_findings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(findings)

    numeric_keys = [
        "mr1_rgb_d_max",
        "mr2_max_d_max",
        "mr3_correct_rgb_d_max",
        "mr4_rgb_d_max",
        "mr5_rgb_d_max",
        "time_s",
    ]
    summary = {
        "n_scenes": len(rows),
        "aborted": aborted,
        "n_findings": len(findings),
        "timing": timing_summary(rows, time.perf_counter() - run_start),
        "stats": summarize(rows, numeric_keys),
        "verdicts": count_verdicts(rows),
        "fault_injection": detection_rates(rows),
        "spearman": correlations(rows),
    }
    with open(out.with_name(out.stem + "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    tm = summary["timing"]
    print(f"\nTiming: total {tm['total_wall_s']:.0f}s; first scene {tm['first_scene_s']:.1f}s (includes CUDA JIT compile)")
    if "rest_median_s" in tm:
        print(
            f"  other scenes: median {tm['rest_median_s']:.3f}s, p99 {tm['rest_p99_s']:.3f}s, max {tm['rest_max_s']:.3f}s"
        )

    print("\nMR1-MR5 verdicts over all scenes:")
    for k in MR_KEYS:
        c: Dict[str, int] = {}
        for r in rows:
            v = r.get(k, "fatal")
            c[v] = c.get(v, 0) + 1
        print(f"  {k}: {c}")
    print("Fault injection (violation = detected):")
    for k, c in summary["fault_injection"].items():
        print(f"  {k}: {c}")
    print(f"Scenes needing review (findings): {len(findings)}/{len(rows)} -> {out.stem}_findings.csv")
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
