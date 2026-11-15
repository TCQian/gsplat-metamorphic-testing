"""One scene through every relation: build the scene and cameras, run MR1-MR5 (and the planted
faults) on all cameras, and return one result row with verdicts and scene properties.

`build_scene_and_cameras` is the one place scenes are built, so the runners and every replay
(diagnosis tools, the undetected-fault study) get identical scenes from the same seed.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import gsplat
import torch

from gsplat_mt.metrics import TOLERANCES, is_power_of_two, verdict
from gsplat_mt.relations.mr1_packed import run_mr1
from gsplat_mt.relations.mr2_batch import run_mr2
from gsplat_mt.relations.mr3_covariance import run_mr3
from gsplat_mt.relations.mr4_world_scale import run_mr4
from gsplat_mt.relations.mr5_permutation import run_mr5
from gsplat_mt.scene import SCENE_MODES, GaussianScene, sample_cameras, sample_scene


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

    `cam0_empty` is kept for old analyses; the relations now render every camera, so
    `all_empty` is what marks a vacuous pass.

    Coverage comes from the rendered alpha; `opaque_frac_*` (alpha > 0.99) shows
    how much of the image is saturated, where hidden Gaussians cannot affect pixels. Visible-Gaussian counts come from
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
    opaque = (alpha > 0.99).float().flatten(1).mean(dim=1)  # [C]
    row: Dict[str, object] = {
        "coverage_cam0": coverage[0].item(),
        "coverage_mean": coverage.mean().item(),
        "opaque_frac_cam0": opaque[0].item(),
        "opaque_frac_mean": opaque.mean().item(),
        "n_empty_cams": int((coverage == 0).sum().item()),
        "coverage_min": coverage.min().item(),  # the least-covered camera
        "n_covered_px": int((alpha > 0).sum().item()),  # over all cameras
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


def build_scene_and_cameras(
    scene_idx: int,
    mode: str,
    args: argparse.Namespace,
    device: str,
):
    """The scene and cameras of one scene index (the single place they are built, so replays match)."""
    generator = torch.Generator(device="cpu").manual_seed(args.seed + scene_idx)

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
        camera_hint=(args.dist_min, args.dist_max, args.fov_deg, args.width / args.height),  # only dense_cover reads it
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


def run_one_scene(
    scene_idx: int,
    mode: str,
    args: argparse.Namespace,
    device: str,
) -> Dict[str, object]:
    seed = args.seed + scene_idx
    scene, cameras = build_scene_and_cameras(scene_idx, mode, args, device)

    row: Dict[str, object] = {"scene_idx": scene_idx, "seed": seed, "mode": mode}
    row.update(scene_properties(scene, cameras))
    stats = render_stats(scene, cameras)
    row.update(stats["row"])

    mr1 = run_mr1(scene, cameras)
    row.update(flatten_diff("mr1_rgb", mr1.rgb, mr1.error))
    row.update(flatten_diff("mr1_alpha", mr1.alpha, mr1.error))
    row["mr1_verdict"] = verdict(
        (mr1.rgb, mr1.alpha), mr1.error, TOLERANCES["mr1"], allow_isolated=True
    )

    if args.mr1_backgrounds:
        mr1_bg = run_mr1(scene, cameras, use_backgrounds=True)
        row.update(flatten_diff("mr1_bg_rgb", mr1_bg.rgb, mr1_bg.error))

    mr2 = run_mr2(scene, cameras)
    if mr2.error is not None:
        row["mr2_max_d_max"] = None
        row["mr2_mean_d_mae"] = None
        row["mr2_mean_d_ssim"] = None
        row["mr2_error"] = mr2.error
    else:
        row["mr2_max_d_max"] = max(d.d_max for d in mr2.per_camera)
        row["mr2_mean_d_mae"] = sum(d.d_mae for d in mr2.per_camera) / len(
            mr2.per_camera
        )
        row["mr2_mean_d_ssim"] = sum(d.d_ssim for d in mr2.per_camera) / len(
            mr2.per_camera
        )
        row["mr2_error"] = None
    row["mr2_verdict"] = verdict(tuple(mr2.per_camera), mr2.error, TOLERANCES["mr2"])

    mr3_correct = run_mr3(scene, cameras, mode="correct")
    row.update(flatten_diff("mr3_correct_rgb", mr3_correct.rgb, mr3_correct.error))
    row["mr3_correct_verdict"] = verdict(
        (mr3_correct.rgb, mr3_correct.alpha), mr3_correct.error, TOLERANCES["mr3"]
    )

    mr3_bug = run_mr3(scene, cameras, mode="xyzw_bug")
    row.update(flatten_diff("mr3_xyzwbug_rgb", mr3_bug.rgb, mr3_bug.error))
    # Fault injection: "violation" here means the injected bug was detected.
    row["mr3_xyzwbug_verdict"] = verdict(
        (mr3_bug.rgb, mr3_bug.alpha), mr3_bug.error, TOLERANCES["mr3"]
    )

    # MR4: sweep world-scale factors; record the worst one.
    mr4_runs = [(k, run_mr4(scene, cameras, k)) for k in args.world_scales]
    mr4_diffs = tuple(d for _, r in mr4_runs for d in (r.rgb, r.alpha))
    mr4_error = next((r.error for _, r in mr4_runs if r.error), None)
    ok = [(k, r) for k, r in mr4_runs if r.rgb is not None]
    worst_k, worst = max(ok, key=lambda kr: kr[1].rgb.d_max) if ok else (None, None)
    row.update(flatten_diff("mr4_rgb", worst.rgb if worst else None, mr4_error))
    row["mr4_worst_k"] = worst_k
    row["mr4_verdict"] = verdict(
        mr4_diffs, mr4_error, TOLERANCES["mr4"], allow_isolated=True
    )

    # Power-of-two k scale exactly in binary, so those renders must be identical:
    # no tolerance, no isolated/tie relabeling.
    exact = [r for k, r in mr4_runs if is_power_of_two(k)]
    if exact:
        exact_diffs = tuple(d for r in exact for d in (r.rgb, r.alpha))
        exact_error = next((r.error for r in exact if r.error), None)
        row["mr4_exact_d_max"] = (
            max(d.d_max for d in exact_diffs if d is not None)
            if any(d is not None for d in exact_diffs)
            else None
        )
        row["mr4_exact_verdict"] = verdict(exact_diffs, exact_error, 0.0)

    # Fault injection: camera translation not scaled, at the largest factor.
    mr4_bug = run_mr4(scene, cameras, max(args.world_scales), mode="no_translation")
    row.update(flatten_diff("mr4_bug_rgb", mr4_bug.rgb, mr4_bug.error))
    row["mr4_bug_verdict"] = verdict(
        (mr4_bug.rgb, mr4_bug.alpha), mr4_bug.error, TOLERANCES["mr4"]
    )

    mr5 = run_mr5(scene, cameras, seed)
    row.update(flatten_diff("mr5_rgb", mr5.rgb, mr5.error))
    row["mr5_depth_tie"] = mr5.depth_tie
    row["mr5_verdict"] = verdict((mr5.rgb, mr5.alpha), mr5.error, TOLERANCES["mr5"])
    if mr5.depth_tie and row["mr5_verdict"] == "violation":
        row["mr5_verdict"] = "tie_ambiguous"  # compositing order of tied depths is ambiguous
    # MR4 rounds every depth, so near-equal depths can swap order between the two renders.
    # Only a violation: an isolated one is already bounded to a single cutoff flip.
    if mr5.depth_tie and row["mr4_verdict"] == "violation":
        row["mr4_verdict"] = "tie_ambiguous"

    # Fault injection: colors not permuted along with the other attributes.
    mr5_bug = run_mr5(scene, cameras, seed, mode="misaligned")
    row.update(flatten_diff("mr5_bug_rgb", mr5_bug.rgb, mr5_bug.error))
    row["mr5_bug_verdict"] = verdict(
        (mr5_bug.rgb, mr5_bug.alpha), mr5_bug.error, TOLERANCES["mr5"]
    )

    # Every relation renders every camera, so a pass proves nothing only if every camera is empty.
    row["relations_cameras"] = "all"  # earlier runs rendered cameras[0] only (MR2 aside)
    flag_vacuous(row, VERDICT_KEYS, stats.get("all_empty", False))

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
    "mr4_exact_verdict",
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


def build_argparser(description: str = None) -> argparse.ArgumentParser:
    """Arguments of the preliminary runner. Its defaults are also the per-scene settings that
    `sampling.scene_args` starts from, so the large-scale runner and replays share them."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-scenes", type=int, default=20, help="scenes per mode")
    p.add_argument(
        "--modes",
        nargs="+",
        default=list(SCENE_MODES),
        choices=SCENE_MODES,
        help="scene types to run (default: all nine)",
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
        default=[0.01, 0.0078125, 0.25, 4.0, 100.0, 128.0],
        help="MR4 world-scale factors k; powers of two are also checked for exact equality",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="results/preliminary.csv")
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
