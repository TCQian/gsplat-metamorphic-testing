"""Experiments on the open items left by the 100,000-scene all-camera run (README_week13.md).

Subcommands (each replays scenes of a run_large_scale.py run built with --seed and all nine modes):

  violations    Bisect the MR1 / MR4 `violation` scenes at the camera that shows the problem
                (diagnose_mr4.py and diagnose_mr1.py look at camera 0 only). For each: the
                per-camera differences, the smallest Gaussian set that reproduces the worst
                pixel, the CLASS line, the conic condition number of the kept Gaussians, and
                whether removing them leaves anything over tolerance.
  derangement   For the undetected MR5 planted faults (study_undetected_faults.py output), re-run
                the fault with a shuffle that has no fixed points. If the fixed-point cause is
                right, every `visible_all_fixed_points` scene is detected.
  nothing       For `affected_gaussians_contribute_nothing` scenes: does each affected Gaussian
                reach a pixel when rendered alone (then it is hidden by others), or not even
                alone (below gsplat's 1/255 alpha cutoff, or off screen)?
  coverage      Why `dense_cover` covers about 44% of pixels and not the predicted 85-98%: coverage
                with Gaussian scales x1, x2, x3, the share of Gaussians surviving culling, and the
                real screen radius against the radius assumed by dense_cover_params.

Usage:
    python gsplat_mt/experiments_open_items.py violations --csv results/allcam_100000.csv --seed 2000
    python gsplat_mt/experiments_open_items.py derangement --undetected results/undetected_allcam_100000.csv --seed 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List

import torch
from diagnose_mr1 import render_mode
from diagnose_mr4 import TOL, bisect, subset
from diagnose_mr4 import render as render_scaled
from metrics import TOLERANCES, verdict
from mr5_permutation import run_mr5
from run_large_scale import sample_config, scene_args
from run_preliminary import build_scene_and_cameras
from scene import SCENE_MODES
from study_undetected_faults import render, visible_mask

ISOLATED_MAX_D_MAX = 4e-3
WORLD_SCALES = [0.01, 0.0078125, 0.25, 4.0, 100.0, 128.0]


def rebuild(i: int, seed: int, device: str = "cuda"):
    """(cfg, args, scene, cameras) of scene i of a run_large_scale.py run with all nine modes."""
    la = argparse.Namespace(
        seed=seed, n_min=10, n_max=5000, max_cameras=8, modes=list(SCENE_MODES), world_scales=WORLD_SCALES
    )
    cfg = sample_config(i, la)
    ns = scene_args(cfg, la)
    scene, cams = build_scene_and_cameras(i, cfg["mode"], ns, device)
    return cfg, ns, scene, cams


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def condition_numbers(scene, keep, cam) -> List[float]:
    """Condition number (larger over smaller eigenvalue) of each kept Gaussian's 2D conic in `cam`."""
    sub = subset(scene, torch.tensor(keep, device=scene.means.device))
    _, _, meta = render_scaled(sub, cam, 1.0, with_meta=True)
    out = []
    for a, b, c in meta["conics"][0].tolist():
        tr, det = a + c, a * c - b * b
        disc = max((tr / 2) ** 2 - det, 0.0) ** 0.5
        lo = tr / 2 - disc
        out.append((tr / 2 + disc) / lo if lo > 0 else float("nan"))
    return out


def camera_table(cams, ra, rb) -> List[Dict[str, float]]:
    rows = []
    for c, cam in enumerate(cams):
        rgb_a, alpha_a = ra(cam)
        rgb_b, alpha_b = rb(cam)
        d = (rgb_b - rgb_a).abs().amax(-1)
        rows.append(
            {
                "camera": c,
                "d_max": d.max().item(),
                "px_over": int((d > TOL).sum()),
                "alpha_d_max": (alpha_b - alpha_a).abs().max().item(),
                "alpha_px_over": int(((alpha_b - alpha_a).abs() > TOL).sum()),
            }
        )
    return rows


def pick_camera(table) -> int:
    """The camera whose image most plausibly made the verdict: the largest D_max if it is above the
    cutoff bound, else the most pixels over tolerance."""
    top = max(table, key=lambda r: r["d_max"])
    if top["d_max"] > ISOLATED_MAX_D_MAX:
        return top["camera"]
    return max(table, key=lambda r: (r["px_over"], r["d_max"]))["camera"]


def violations(a) -> Dict[str, object]:
    rows = read_rows(a.csv)
    cases = [("mr1", r) for r in rows if r.get("mr1_verdict") == "violation"] + [
        ("mr4", r) for r in rows if r.get("mr4_verdict") == "violation"
    ]
    print(f"{len(cases)} violation scenes")
    results = []
    for rel, r in cases:
        i = int(r["scene_idx"])
        cfg, ns, scene, cams = rebuild(i, a.seed)
        if cfg["mode"] != r["mode"]:
            print(f"scene {i}: mode {cfg['mode']} != CSV {r['mode']}; arguments do not match, skipping")
            continue
        k = float(r["mr4_worst_k"]) if rel == "mr4" else 1.0
        if rel == "mr4":
            ra = lambda cam: render_scaled(scene, cam, 1.0)  # noqa: E731
            rb = lambda cam: render_scaled(scene, cam, k)  # noqa: E731
        else:
            ra = lambda cam: render_mode(cam, False)(scene, False)  # noqa: E731
            rb = lambda cam: render_mode(cam, True)(scene, False)  # noqa: E731
        table = camera_table(cams, ra, rb)
        c = pick_camera(table)
        cam = cams[c]
        print(f"\n=== {rel.upper()} scene {i} [{cfg['mode']}] N={scene.n} cameras={len(cams)} "
              f"{cam.width}x{cam.height} k={k:g} depth_tie={r['mr5_depth_tie']}")
        for t in table:
            print(f"  camera {t['camera']}: D_max {t['d_max']:.2e}, {t['px_over']} px over {TOL:g}, "
                  f"alpha D_max {t['alpha_d_max']:.2e}, {t['alpha_px_over']} alpha px over")
        print(f"  bisecting camera {c}")
        kw = {}
        if rel == "mr1":
            kw = dict(render_a=render_mode(cam, False), render_b=render_mode(cam, True), k_b=1.0)
        res = bisect(scene, cam, k, by="color", **kw)
        entry: Dict[str, object] = {
            "relation": rel, "scene_idx": i, "mode": cfg["mode"], "n_gaussians": scene.n,
            "k": k, "camera": c, "cameras": table, "depth_tie": r["mr5_depth_tie"] == "True",
        }
        if res:
            kappas = condition_numbers(scene, res["keep"], cam)
            cls = res["class"]
            if cls == "unclassified" and res["n_keep"] == 1:
                cls = "single Gaussian, T unchanged (smooth conic difference?)"
            rest = [g for g in range(scene.n) if g not in set(res["keep"])]
            sub = subset(scene, torch.tensor(rest, device=scene.means.device))
            if rel == "mr4":
                ra2, _ = render_scaled(sub, cam, 1.0)
                rb2, _ = render_scaled(sub, cam, k)
            else:
                ra2, _ = render_mode(cam, False)(sub, False)
                rb2, _ = render_mode(cam, True)(sub, False)
            d2 = (rb2 - ra2).abs().amax(-1)
            print(f"  conic condition number of kept Gaussians: {['%.3g' % x for x in kappas]}")
            print(f"  without the kept Gaussians: {int((d2 > TOL).sum())} px over {TOL:g}, D_max {d2.max().item():.2e}")
            entry.update(
                {"class": cls, "n_keep": res["n_keep"], "rel_t_change": res["rel_t"], "min_depth_gap": res["min_gap"],
                 "pixel": list(res["pixel"]), "kappa_kept": kappas,
                 "px_over_without_kept": int((d2 > TOL).sum()), "d_max_without_kept": d2.max().item()}
            )
        else:
            entry["class"] = "nothing over tolerance on the chosen camera"
        results.append(entry)
    print("\nSUMMARY")
    for e in results:
        print(f"  {e['relation']} scene {e['scene_idx']} [{e['mode']}] camera {e['camera']}: {e['class']}"
              + (f" ({e['n_keep']} kept)" if "n_keep" in e else ""))
    return {"violations": results}


def derangement(a) -> Dict[str, object]:
    rows = [r for r in read_rows(a.undetected) if r.get("fault") == "mr5_bug"]
    print(f"{len(rows)} undetected MR5 scenes")
    out, counts = [], {}
    for r in rows:
        i = int(r["scene_idx"])
        cfg, ns, scene, cams = rebuild(i, a.seed)
        seed, n, dev = ns.seed + i, scene.n, scene.means.device
        q = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(seed))
        perm = torch.empty(n, dtype=torch.long)
        perm[q] = q.roll(-1)  # perm[q[j]] = q[j+1]: a single n-cycle, no fixed point (n >= 2)
        assert bool((perm != torch.arange(n)).all())
        perm = perm.to(dev)

        def detected(res) -> bool:
            return res.error is None and verdict((res.rgb, res.alpha), None, TOLERANCES["mr5"]) == "violation"

        orig = detected(run_mr5(scene, cams, seed, mode="misaligned"))
        der = detected(run_mr5(scene, cams, seed, mode="misaligned", perm=perm))
        ok = run_mr5(scene, cams, seed, mode="correct", perm=perm)
        correct_v = verdict((ok.rgb, ok.alpha), ok.error, TOLERANCES["mr5"])
        row = {"scene_idx": i, "mode": cfg["mode"], "n_gaussians": n, "cause": r["cause"],
               "detected_random_shuffle": orig, "detected_derangement": der,
               "correct_relation_with_derangement": correct_v}
        out.append(row)
        key = (r["cause"], der)
        counts[key] = counts.get(key, 0) + 1
        print(f"  scene {i} [{cfg['mode']}] N={n} cause={r['cause']}: random shuffle detected={orig}, "
              f"derangement detected={der}, correct relation under derangement: {correct_v}")
    summary = {f"{c} / derangement detects={d}": n for (c, d), n in sorted(counts.items(), key=str)}
    print("\nSUMMARY", json.dumps(summary, indent=2))
    return {"derangement": out, "summary": summary}


def nothing(a) -> Dict[str, object]:
    rows = [r for r in read_rows(a.undetected)
            if r.get("fault") == "mr5_bug" and r["cause"] == "affected_gaussians_contribute_nothing"]
    print(f"{len(rows)} contribute-nothing scenes")
    out = []
    for r in rows:
        i = int(r["scene_idx"])
        cfg, ns, scene, cams = rebuild(i, a.seed)
        seed, n, dev = ns.seed + i, scene.n, scene.means.device
        perm = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(seed)).to(dev)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(n, device=dev)
        affected = visible_mask(scene, cams) & (scene.colors[inv] != scene.colors).any(dim=-1)
        idx = affected.nonzero().flatten().tolist()
        print(f"\nscene {i} [{cfg['mode']}] N={n}: {len(idx)} affected Gaussians")
        per = []
        for g in idx[:20]:
            _, alpha_alone, _ = render(subset(scene, torch.tensor([g], device=dev)), cams)
            _, alpha_full, _ = render(scene, cams)
            reach = float(alpha_alone.max())
            verdict_g = "reaches a pixel alone, so hidden by other Gaussians" if reach > 0 else (
                "never reaches a pixel even alone (below the 1/255 cutoff or off screen)")
            print(f"  Gaussian {g}: opacity {scene.opacities[g].item():.4g}, max alpha alone {reach:.3g}: {verdict_g}")
            per.append({"gaussian": g, "opacity": scene.opacities[g].item(), "max_alpha_alone": reach,
                        "reaches_pixel_alone": reach > 0})
        out.append({"scene_idx": i, "mode": cfg["mode"], "n_affected": len(idx), "affected": per})
    return {"nothing": out}


def coverage(a) -> Dict[str, object]:
    rows = [r for r in read_rows(a.csv) if r["mode"] == "dense_cover"][: a.n]
    print(f"{len(rows)} dense_cover scenes")
    out = []
    for r in rows:
        i = int(r["scene_idx"])
        cfg, ns, scene, cams = rebuild(i, a.seed)
        rec: Dict[str, object] = {"scene_idx": i, "n_gaussians": scene.n, "csv_coverage_mean": float(r["coverage_mean"])}
        for m in (1.0, 2.0, 3.0):
            s2 = subset(scene, torch.arange(scene.n, device=scene.means.device))
            s2.scales = scene.scales * m
            _, alpha, meta = render(s2, cams)
            rec[f"coverage_x{m:g}"] = float((alpha > 0).float().mean())
            if m == 1.0:
                radii = meta["radii"].float()  # [C,N,2] pixels
                vis = (radii > 0).all(dim=-1)
                rec["visible_share"] = float(vis.any(dim=0).float().mean())
                f = torch.stack([c.K[0, 0] for c in cams])[:, None]  # [C,1]
                z = meta["depths"]  # [C,N]
                pred = 2.5 * scene.scales.mean(dim=-1)[None] * f / z  # radius assumed by dense_cover_params, pixels
                ok = vis & (z > 0)
                # gsplat's radii bound the footprint (about 3 sigma of the longest axis), so this ratio
                # is a rough size check, not the footprint radius itself.
                rec["radius_ratio_median"] = float((radii.mean(dim=-1)[ok] / pred[ok]).median())
        out.append(rec)
        print(f"  scene {i}: N={scene.n} coverage x1 {rec['coverage_x1']:.2f} (CSV {rec['csv_coverage_mean']:.2f}), "
              f"x2 {rec['coverage_x2']:.2f}, x3 {rec['coverage_x3']:.2f}; visible {rec['visible_share']:.2f}; "
              f"screen radius / assumed {rec['radius_ratio_median']:.2f}")
    med = lambda k: statistics.median(r[k] for r in out)  # noqa: E731
    summary = {k: med(k) for k in ("csv_coverage_mean", "coverage_x1", "coverage_x2", "coverage_x3",
                                    "visible_share", "radius_ratio_median")}
    print("\nSUMMARY (medians)", json.dumps(summary, indent=2))
    return {"coverage": out, "summary": summary}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("violations", "derangement", "nothing", "coverage"):
        s = sub.add_parser(name)
        s.add_argument("--seed", type=int, default=2000)
        s.add_argument("--out", default=None, help="JSON output (default results/open_<cmd>.json)")
        if name in ("violations", "coverage"):
            s.add_argument("--csv", required=True)
        else:
            s.add_argument("--undetected", required=True, help="study_undetected_faults.py CSV")
        if name == "coverage":
            s.add_argument("--n", type=int, default=60)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is required (gsplat has no CPU backend).")
    result = {"violations": violations, "derangement": derangement, "nothing": nothing, "coverage": coverage}[a.cmd](a)
    out = Path(a.out or f"results/open_{a.cmd}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
