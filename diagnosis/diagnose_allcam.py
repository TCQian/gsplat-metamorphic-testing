"""Follow-up diagnoses of the all-camera run (runners/run_large_scale.py, seed 2000, nine scene types).

Subcommands (each replays scenes of that run; --seed and --modes select another run):

  violations    Bisect the MR1 / MR4 `violation` scenes at the camera that shows the problem
                (as diagnose_mr4.py --camera worst does). For each: the
                per-camera differences, the smallest Gaussian set that reproduces the worst
                pixel, the CLASS line, the conic condition number of the kept Gaussians, and
                whether removing them leaves anything over tolerance.
  pixels        For the same violation scenes, account for every pixel over tolerance in every
                camera (and, for MR4, at every inexact factor k): bisect the worst pixel not yet
                accounted for, count every over-tolerance pixel that the kept Gaussians alone
                reproduce to at least half its difference as accounted for, and repeat. Then
                remove all kept Gaussians and count what is still over tolerance.
  derangement   For the undetected MR5 planted faults (study_undetected_faults.py output), re-run
                the fault with the random shuffle of the earlier runs and with the derangement
                MR5 now uses. If the fixed-point cause is right, the derangement detects every
                `visible_all_fixed_points` scene.
  nothing       For `affected_gaussians_contribute_nothing` scenes: does each affected Gaussian
                reach a pixel when rendered alone (then it is hidden by others), or not even
                alone (below gsplat's 1/255 alpha cutoff, or off screen)?
  coverage      Why `dense_cover` covers about 44% of pixels and not the predicted 85-98%: coverage
                with Gaussian scales x1, x2, x3, the share of Gaussians surviving culling, and the
                real screen radius against the radius assumed by dense_cover_params.

Each writes results/open_<subcommand>.json (or --out).

Usage (from the repository root):
    python diagnosis/diagnose_allcam.py violations --csv results/allcam_100000.csv
    python diagnosis/diagnose_allcam.py pixels --csv results/allcam_100000.csv
    python diagnosis/diagnose_allcam.py derangement --undetected results/undetected_allcam_100000.csv
    python diagnosis/diagnose_allcam.py nothing --undetected results/undetected_allcam_100000.csv
    python diagnosis/diagnose_allcam.py coverage --csv results/allcam_100000.csv --n 60
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, for `import gsplat_mt`

from diagnosis.diagnose_mr1 import render_mode
from diagnosis.diagnose_mr4 import TOL, bisect, camera_table, condition_numbers, pick_camera, subset
from diagnosis.diagnose_mr4 import render as render_scaled
from diagnosis.study_undetected_faults import render, visible_mask
from gsplat_mt.metrics import TOLERANCES, is_power_of_two, verdict
from gsplat_mt.relations.mr5_permutation import derangement as derangement_perm
from gsplat_mt.relations.mr5_permutation import random_shuffle, run_mr5
from gsplat_mt.sampling import WORLD_SCALES, add_run_args, replay_scene, run_namespace


def rebuild(i: int, a):
    """(cfg, args, scene, cameras) of scene i of the run selected by a.seed and a.modes."""
    return replay_scene(i, run_namespace(a.seed, a.modes))


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def violations(a) -> Dict[str, object]:
    rows = read_rows(a.csv)
    cases = [("mr1", r) for r in rows if r.get("mr1_verdict") == "violation"] + [
        ("mr4", r) for r in rows if r.get("mr4_verdict") == "violation"
    ]
    print(f"{len(cases)} violation scenes")
    results = []
    for rel, r in cases:
        i = int(r["scene_idx"])
        cfg, ns, scene, cams = rebuild(i, a)
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


def explain_image(scene, cam, relation, k, max_rounds) -> Dict[str, object]:
    """Account for every over-tolerance pixel of one camera image of an MR1 or MR4 comparison."""
    if relation == "mr1":
        ra_s, rb_s = render_mode(cam, False), render_mode(cam, True)
        kw = dict(render_a=ra_s, render_b=rb_s, k_b=1.0)
    else:
        ra_s = lambda sc, meta: render_scaled(sc, cam, 1.0, with_meta=meta)  # noqa: E731
        rb_s = lambda sc, meta: render_scaled(sc, cam, k, with_meta=meta)  # noqa: E731
        kw = {}

    def diffs(sc):
        rgb_a, al_a = ra_s(sc, False)
        rgb_b, al_b = rb_s(sc, False)
        return (rgb_b - rgb_a).abs().amax(-1), (al_b - al_a).abs()

    dev = scene.means.device
    d_rgb, d_al = diffs(scene)
    over_rgb, over_al = d_rgb > TOL, d_al > TOL
    over = over_rgb | over_al
    left = over.clone()
    width = d_rgb.shape[1]
    rounds, kept_all = [], set()
    while left.any() and len(rounds) < max_rounds:
        cand = torch.where(left & over_rgb, d_rgb, torch.zeros_like(d_rgb))
        if cand.max() > 0:
            by, flat = "color", int(cand.flatten().argmax())
        else:
            by, flat = "alpha", int(torch.where(left, d_al, torch.zeros_like(d_al)).flatten().argmax())
        y, x = divmod(flat, width)
        res = bisect(scene, cam, k, by=by, pixel=(y, x), **kw)
        if res is None:
            left[y, x] = False
            rounds.append({"pixel": [y, x], "by": by, "pattern": "not reproduced", "n_explained": 1})
            continue
        keep = res["keep"]
        kept_all.update(keep)
        s_rgb, s_al = diffs(subset(scene, torch.tensor(keep, device=dev)))
        # the same criterion as the bisection: the kept Gaussians alone give at least half the difference
        expl = (over_rgb & (s_rgb >= 0.5 * d_rgb)) | (~over_rgb & over_al & (s_al >= 0.5 * d_al))
        expl[y, x] = True
        newly = expl & left
        left &= ~expl
        rounds.append({"pixel": [y, x], "by": by, "pattern": res["pattern"], "class": res["class"],
                       "n_keep": res["n_keep"], "keep": keep[:20], "kappa": res["kappa"][:6], "diff": res["diff"],
                       "min_t": res["min_t"], "n_explained": int(newly.sum())})
    rest = [g for g in range(scene.n) if g not in kept_all]
    if rest:
        r_rgb, r_al = diffs(subset(scene, torch.tensor(rest, device=dev)))
        after = (r_rgb > TOL) | (r_al > TOL)
    else:
        after = torch.zeros_like(over)
    return {
        "n_over": int(over.sum()),
        "d_max": float(torch.maximum(d_rgb, d_al).max()),
        "rounds": rounds,
        "n_unexplained": int(left.sum()),
        "over_after_removing_kept": int(after.sum()),
        "over_only_after_removing_kept": int((after & ~over).sum()),
    }


def pixels(a) -> Dict[str, object]:
    rows = read_rows(a.csv)
    cases = [("mr1", r) for r in rows if r.get("mr1_verdict") == "violation"] + [
        ("mr4", r) for r in rows if r.get("mr4_verdict") == "violation"
    ]
    inexact = [k for k in WORLD_SCALES if not is_power_of_two(k)]  # power-of-two factors are exactly 0
    print(f"{len(cases)} violation scenes")
    results = []
    for rel, r in cases:
        i = int(r["scene_idx"])
        cfg, ns, scene, cams = rebuild(i, a)
        if cfg["mode"] != r["mode"]:
            print(f"scene {i}: mode {cfg['mode']} != CSV {r['mode']}; arguments do not match, skipping")
            continue
        images = []
        for k in ([1.0] if rel == "mr1" else inexact):
            if rel == "mr4":
                ra = lambda cam, k=k: render_scaled(scene, cam, 1.0)  # noqa: E731
                rb = lambda cam, k=k: render_scaled(scene, cam, k)  # noqa: E731
            else:
                ra = lambda cam: render_mode(cam, False)(scene, False)  # noqa: E731
                rb = lambda cam: render_mode(cam, True)(scene, False)  # noqa: E731
            for t in camera_table(cams, ra, rb):
                if not (t["px_over"] or t["alpha_px_over"]):
                    continue
                c = t["camera"]
                print(f"\n=== {rel.upper()} scene {i} [{cfg['mode']}] N={scene.n} camera {c} k={k:g}: "
                      f"{t['px_over']} rgb / {t['alpha_px_over']} alpha px over {TOL:g}, D_max {t['d_max']:.2e}")
                img = explain_image(scene, cams[c], rel, k, a.max_rounds)
                img.update(camera=c, k=k)
                images.append(img)
                print(f"  -> {img['n_over']} px over tolerance; {len(img['rounds'])} bisection(s): "
                      + "; ".join(f"{q['pattern']} ({q.get('n_keep', 0)} kept) accounts for {q['n_explained']} px"
                                  for q in img["rounds"])
                      + f"; {img['n_unexplained']} px not accounted for; after removing every kept Gaussian "
                      f"{img['over_after_removing_kept']} px over ({img['over_only_after_removing_kept']} not over before)")
        results.append({"relation": rel, "scene_idx": i, "mode": cfg["mode"], "n_gaussians": scene.n, "images": images})
    print("\nSUMMARY")
    tot = {"images": 0, "px_over": 0, "px_unexplained": 0}
    patterns: Dict[str, int] = {}
    for e in results:
        for img in e["images"]:
            tot["images"] += 1
            tot["px_over"] += img["n_over"]
            tot["px_unexplained"] += img["n_unexplained"]
            for q in img["rounds"]:
                patterns[q["pattern"]] = patterns.get(q["pattern"], 0) + q["n_explained"]
        print(f"  {e['relation']} scene {e['scene_idx']}: " + ", ".join(
            f"camera {img['camera']} k={img['k']:g}: {img['n_over']} px, "
            + "/".join(sorted({q['pattern'] for q in img['rounds']})) for img in e["images"]))
    print(f"  {tot}; pixels per pattern: {patterns}")
    return {"pixels": results, "summary": {**tot, "pixels_per_pattern": patterns}}


def derangement(a) -> Dict[str, object]:
    rows = [r for r in read_rows(a.undetected) if r.get("fault") == "mr5_bug"]
    print(f"{len(rows)} undetected MR5 scenes")
    out, counts = [], {}
    for r in rows:
        i = int(r["scene_idx"])
        cfg, ns, scene, cams = rebuild(i, a)
        seed, n = ns.seed + i, scene.n
        rnd = random_shuffle(n, seed)  # the shuffle of the runs before MR5 used a derangement

        def detected(res) -> bool:
            return res.error is None and verdict((res.rgb, res.alpha), None, TOLERANCES["mr5"]) == "violation"

        orig = detected(run_mr5(scene, cams, seed, mode="misaligned", perm=rnd))
        der = detected(run_mr5(scene, cams, seed, mode="misaligned"))  # default: derangement
        ok = run_mr5(scene, cams, seed, mode="correct")
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
    shuffle = random_shuffle if a.shuffle == "random" else derangement_perm  # the shuffle of the run studied
    out = []
    for r in rows:
        i = int(r["scene_idx"])
        cfg, ns, scene, cams = rebuild(i, a)
        seed, n, dev = ns.seed + i, scene.n, scene.means.device
        perm = shuffle(n, seed).to(dev)
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
        cfg, ns, scene, cams = rebuild(i, a)
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
    for name in ("violations", "pixels", "derangement", "nothing", "coverage"):
        s = sub.add_parser(name)
        add_run_args(s)
        s.add_argument("--out", default=None, help="JSON output (default results/open_<cmd>.json)")
        if name in ("violations", "pixels", "coverage"):
            s.add_argument("--csv", required=True)
        else:
            s.add_argument("--undetected", required=True, help="study_undetected_faults.py CSV")
        if name == "nothing":
            s.add_argument("--shuffle", choices=["derangement", "random"], default="derangement",
                           help="MR5 shuffle of the run studied (random for runs before MR5 used a derangement)")
        if name == "coverage":
            s.add_argument("--n", type=int, default=60)
        if name == "pixels":
            s.add_argument("--max-rounds", type=int, default=20, help="bisections per camera image at most")
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is required (gsplat has no CPU backend).")
    result = {"violations": violations, "pixels": pixels, "derangement": derangement, "nothing": nothing,
              "coverage": coverage}[a.cmd](a)
    out = Path(a.out or f"results/open_{a.cmd}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
