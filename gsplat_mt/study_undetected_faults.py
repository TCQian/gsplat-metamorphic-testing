"""Why were some planted faults not detected? Replays those scenes and tests the cause.

A planted fault (MR3 `xyzw_bug`, MR4 `no_translation`, MR5 `misaligned`) is "undetected"
when its verdict is `pass` on a non-empty image. For each such scene of a
run_large_scale.py CSV this script rebuilds the scene and, by experiment, assigns a cause.

Causes (checked in this order):
  caught_by_other_cameras   the fault is detected when every camera is rendered, but camera 0
                            alone missed it (only meaningful on CSVs from before all-camera runs)
  single_gaussian           MR5: one Gaussian, so a shuffle is the identity
  visible_all_fixed_points  MR5: every visible Gaussian stays at its index under the shuffle, so
                            its color is unchanged; re-running with other shuffles shows whether
                            another shuffle would have caught the fault (`alt_detect_rate`)
  visible_colors_unchanged  MR5: visible Gaussians move, but swapped colors are identical
  visible_isotropic         MR3: every visible Gaussian has equal scales, so rotation is irrelevant
  *_contribute_nothing      the Gaussians the fault touches are visible to gsplat's culling but
                            removing them (opacity 0) does not change the image: they never
                            reach a pixel (below the 1/255 alpha cutoff or fully occluded)
  unexplained               none of the above; needs a manual look

`depth_tie` is recorded for each scene, so the causes can be counted with and without scenes
that have near-equal depths (where the blending order is ambiguous).

Usage (the CSV's own sampling arguments must be repeated if they were not the defaults):
    python gsplat_mt/study_undetected_faults.py --csv results/xl_scale.csv --seed 1000 \\
        --faults mr5_bug --out results/undetected_xl.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import gsplat
import torch
from metrics import TOLERANCES, verdict
from mr3_covariance import run_mr3
from mr4_world_scale import run_mr4
from mr5_permutation import has_depth_tie, run_mr5
from run_large_scale import sample_config, scene_args
from run_preliminary import build_scene_and_cameras
from scene import DEFAULT_MODES, SCENE_MODES, stack_cameras

NEAR, FAR = 0.01, 100.0
FAULTS = ("mr5_bug", "mr3_xyzwbug", "mr4_bug")
ALT_SEED_BASE = 10**6


def render(scene, cameras, opacities=None):
    viewmats, Ks, width, height = stack_cameras(cameras)
    rgb, alpha, meta = gsplat.rasterization(
        means=scene.means,
        quats=scene.quats,
        scales=scene.scales,
        opacities=scene.opacities if opacities is None else opacities,
        colors=scene.colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=NEAR,
        far_plane=FAR,
        sh_degree=None,
        packed=False,
    )
    return rgb, alpha, meta


def visible_mask(scene, cameras) -> torch.Tensor:
    """Gaussians gsplat does not cull in at least one camera, [N] bool."""
    _, _, meta = render(scene, cameras)
    radii = meta["radii"]
    vis = (radii > 0).all(dim=-1) if radii.dim() == 3 else radii > 0  # [C,N]
    return vis.any(dim=0)


def contributes_nothing(scene, cameras, affected: torch.Tensor) -> bool:
    """True if zeroing the opacity of `affected` Gaussians leaves every pixel unchanged."""
    rgb_a, alpha_a, _ = render(scene, cameras)
    opacities = scene.opacities.clone()
    opacities[affected] = 0.0
    rgb_b, alpha_b, _ = render(scene, cameras, opacities)
    return bool((rgb_a == rgb_b).all() and (alpha_a == alpha_b).all())


def run_fault(fault: str, scene, cameras, seed: int, k: float):
    """(detected, d_max) for the planted fault on `cameras`."""
    if fault == "mr5_bug":
        r, kind = run_mr5(scene, cameras, seed, mode="misaligned"), "mr5"
    elif fault == "mr3_xyzwbug":
        r, kind = run_mr3(scene, cameras, mode="xyzw_bug"), "mr3"
    elif fault == "mr4_bug":
        r, kind = run_mr4(scene, cameras, k, mode="no_translation"), "mr4"
    else:
        raise ValueError(fault)
    if r.error is not None:
        return False, None
    det = verdict((r.rgb, r.alpha), None, TOLERANCES[kind]) == "violation"
    return det, max(r.rgb.d_max, r.alpha.d_max)


def classify_mr5(scene, cameras, seed: int, n_alt: int) -> Dict[str, object]:
    n, dev = scene.n, scene.means.device
    info: Dict[str, object] = {}
    if n == 1:
        return {"cause": "single_gaussian"}
    perm = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(seed)).to(dev)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n, device=dev)
    vis = visible_mask(scene, cameras)
    moved = vis & (perm != torch.arange(n, device=dev))
    # Under the fault, Gaussian g receives the color stored at index inv[g].
    changed = (scene.colors[inv] != scene.colors).any(dim=-1)
    affected = vis & changed
    info.update(
        n_visible=int(vis.sum()),
        n_visible_moved=int(moved.sum()),
        n_visible_color_changed=int(affected.sum()),
    )
    if not moved.any():
        # Would another shuffle have caught it? Count detections over alternative seeds.
        hits = sum(
            run_fault("mr5_bug", scene, cameras, ALT_SEED_BASE + s, 0.0)[0] for s in range(n_alt)
        )
        info.update(cause="visible_all_fixed_points", alt_detect_rate=hits / max(n_alt, 1))
    elif not affected.any():
        info["cause"] = "visible_colors_unchanged"
    elif contributes_nothing(scene, cameras, affected):
        info["cause"] = "affected_gaussians_contribute_nothing"
    else:
        info["cause"] = "unexplained"
    return info


def classify_mr3(scene, cameras) -> Dict[str, object]:
    vis = visible_mask(scene, cameras)
    ratio = scene.scales.amax(dim=-1) / scene.scales.amin(dim=-1).clamp_min(1e-12)
    aniso = vis & (ratio > 1.001)  # rotation only matters for non-isotropic Gaussians
    info = {"n_visible": int(vis.sum()), "n_visible_anisotropic": int(aniso.sum())}
    if not aniso.any():
        info["cause"] = "visible_isotropic"
    elif contributes_nothing(scene, cameras, aniso):
        info["cause"] = "anisotropic_gaussians_contribute_nothing"
    else:
        info["cause"] = "unexplained"
    return info


def classify_mr4(scene, cameras, k: float) -> Dict[str, object]:
    # Translation left unscaled moves the camera relative to the scene, so an undetected
    # case has no obvious mechanism; record the numbers and flag it for a manual look.
    _, alpha, _ = render(scene, cameras)
    return {"cause": "unexplained", "coverage_mean": float((alpha > 0).float().mean())}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="run_large_scale.py results CSV")
    p.add_argument("--faults", nargs="+", default=list(FAULTS), choices=FAULTS)
    p.add_argument("--skip-modes", nargs="+", default=[], help="e.g. single: trivially undetectable for MR5")
    p.add_argument("--limit", type=int, default=None, help="study at most this many scenes per fault")
    p.add_argument("--alt-seeds", type=int, default=16, help="MR5: extra shuffles to try per scene")
    # the sampling arguments of the run that produced the CSV
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=SCENE_MODES)
    p.add_argument("--n-min", type=int, default=10)
    p.add_argument("--n-max", type=int, default=5000)
    p.add_argument("--max-cameras", type=int, default=8)
    p.add_argument("--world-scales", type=float, nargs="+", default=[0.01, 0.0078125, 0.25, 4.0, 100.0, 128.0])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/undetected_faults.csv")
    args = p.parse_args(argv)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("ERROR: CUDA is required (gsplat has no CPU backend).", file=sys.stderr)
        return 1

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows: List[Dict[str, object]] = []
    for fault in args.faults:
        key = f"{fault}_verdict"
        picked = [
            r for r in rows if r.get(key) == "pass" and r["mode"] not in args.skip_modes
        ]  # undetected on a non-empty image
        if args.limit:
            picked = picked[: args.limit]
        print(f"{fault}: {len(picked)} undetected scenes to study")
        for r in picked:
            i = int(r["scene_idx"])
            cfg = sample_config(i, args)
            if cfg["mode"] != r["mode"]:
                print(f"  scene {i}: sampled mode {cfg['mode']} != CSV mode {r['mode']}; "
                      "the sampling arguments do not match the CSV, skipping", file=sys.stderr)
                continue
            ns = scene_args(cfg, args)
            scene, cams = build_scene_and_cameras(i, cfg["mode"], ns, args.device)
            seed, k = ns.seed + i, max(args.world_scales)

            det_cam0, _ = run_fault(fault, scene, cams[:1], seed, k)
            det_all, d_all = run_fault(fault, scene, cams, seed, k)
            row: Dict[str, object] = {
                "fault": fault, "scene_idx": i, "mode": cfg["mode"], "n_gaussians": scene.n,
                "n_cameras": len(cams), "detected_cam0": det_cam0, "detected_all_cameras": det_all,
                "d_max_all_cameras": d_all,
                "depth_tie": has_depth_tie(scene, cams, NEAR, FAR, 1e-6),
            }
            if det_all:
                row["cause"] = "caught_by_other_cameras" if not det_cam0 else "detected_on_replay"
            elif fault == "mr5_bug":
                row.update(classify_mr5(scene, cams, seed, args.alt_seeds))
            elif fault == "mr3_xyzwbug":
                row.update(classify_mr3(scene, cams))
            else:
                row.update(classify_mr4(scene, cams, k))
            out_rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in out_rows for k in r})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    summary: Dict[str, object] = {}
    for fault in args.faults:
        mine = [r for r in out_rows if r["fault"] == fault]
        no_tie = [r for r in mine if not r["depth_tie"]]
        summary[fault] = {
            "studied": len(mine),
            "causes": dict(Counter(r["cause"] for r in mine)),
            "studied_without_depth_tie": len(no_tie),
            "causes_without_depth_tie": dict(Counter(r["cause"] for r in no_tie)),
        }
        rates = [r["alt_detect_rate"] for r in mine if r.get("alt_detect_rate") is not None]
        if rates:
            summary[fault]["mean_alt_detect_rate_fixed_point_scenes"] = sum(rates) / len(rates)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
