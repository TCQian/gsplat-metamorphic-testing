"""Where do packed and unpacked rasterization first differ? (MR1 cutoff flips)

An MR1 cutoff flip means one Gaussian's alpha at a pixel lands on opposite sides
of gsplat's threshold (1/255) in the two paths. This script finds which stage
produces the different bits. For each scene it:

1. bisects (as diagnose_mr1.py) to the Gaussian(s) behind the worst pixel;
2. renders the whole scene with packed=False and packed=True and compares the
   projection outputs gsplat reports in `meta` (means2d, conics, opacities,
   depths, radii) bit for bit, for the kept Gaussians and for every Gaussian
   visible in both paths (how common last-bit differences are in general);
3. for a single kept Gaussian, recomputes its alpha at the pixel in float32
   from each path's own values, at the pixel center and at the pixel corner,
   and checks which of the two reproduces the blended/skipped pattern of the two
   renders (alpha < 1/255 means skipped).

If the kept Gaussian's projection outputs differ between the paths, the
difference is already there before rasterization. If they are bit-identical,
it arises later. The alpha recomputation uses torch's float32 exp, not the
kernel's, so a value within one rounding step of 1/255 can disagree with the kernel.

Usage:
    python gsplat_mt/diagnose_mr1_inputs.py --scene-idx 186
    python gsplat_mt/diagnose_mr1_inputs.py --scenes-from results/large_scale_w11.csv --where mr1_verdict=isolated_discontinuity
"""

from __future__ import annotations

import argparse
import csv

import torch
from diagnose_mr1 import render_mode
from diagnose_mr4 import bisect, load, subset

KEYS = ("means2d", "conics", "opacities", "depths", "radii")
THRESHOLD = 1.0 / 255.0


def dense(meta, n, packed):
    """Per-Gaussian [N, ...] tensors and a visibility mask, from packed or unpacked meta."""
    if packed:
        ids = meta["gaussian_ids"].long()
        out = {}
        for k in KEYS:
            v = meta[k]
            full = torch.zeros((n,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
            full[ids] = v
            out[k] = full
        vis = torch.zeros(n, dtype=torch.bool, device=ids.device)
        vis[ids] = True
    else:
        out = {k: meta[k][0] for k in KEYS}
        vis = (meta["radii"][0] > 0).all(-1)
    return out, vis


def ordered(x):
    """float32 bits mapped to integers that increase with the value, so |a - b| is the ULP distance."""
    i = x.contiguous().view(torch.int32).long()
    return torch.where(i < 0, -(2**31) - i, i)


def ulps(a, b):
    if a.dtype == torch.float32:
        return (ordered(a) - ordered(b)).abs()
    return (a.long() - b.long()).abs()


def alpha_f32(mean2d, conic, opacity, px, py):
    dx, dy = mean2d[0] - px, mean2d[1] - py
    sigma = 0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) + conic[1] * dx * dy
    return torch.clamp(opacity * torch.exp(-sigma), max=0.999)


def diagnose(a, idx):
    mode, scene, cameras = load(a, idx)
    cam = cameras[0]
    print(f"scene {idx} [{mode}] n={scene.n} res={cam.width}x{cam.height}")
    fu, fp = render_mode(cam, False), render_mode(cam, True)
    res = bisect(scene, cam, 1.0, by="color", render_a=fu, render_b=fp, k_b=1.0)
    if res is None:
        return None
    keep, (y, x) = res["keep"], res["pixel"]
    n = scene.n

    _, _, mu = fu(scene, True)
    _, _, mp = fp(scene, True)
    du, vu = dense(mu, n, False)
    dp, vp = dense(mp, n, True)
    both = vu & vp
    print(
        f"\n  visible in unpacked: {int(vu.sum())}, in packed: {int(vp.sum())}, "
        f"in both: {int(both.sum())}, only one path: {int((vu ^ vp).sum())}"
    )
    print("  projection outputs over the Gaussians visible in both paths (bit-exact comparison):")
    for k in KEYS:
        d = ulps(du[k][both], dp[k][both])
        if d.dim() > 1:
            d = d.amax(-1)
        print(f"    {k:10} differing: {int((d > 0).sum())} of {int(both.sum())}   max distance: {int(d.max()) if d.numel() else 0} ULP")

    any_diff = False
    print(f"\n  kept Gaussians ({len(keep)}), unpacked vs packed:")
    for g in keep[:4]:
        if not (vu[g] and vp[g]):
            print(f"    Gaussian {g}: visible in unpacked={bool(vu[g])}, packed={bool(vp[g])}")
            any_diff = True
            continue
        for k in KEYS:
            u, p = du[k][g], dp[k][g]
            d = int(ulps(u, p).max())
            any_diff |= d > 0 and k != "radii"
            fmt = lambda t: ", ".join(f"{v:.9g}" for v in t.flatten().tolist())
            print(f"    Gaussian {g} {k:10} unpacked=[{fmt(u)}] packed=[{fmt(p)}]  {d} ULP")

    sub = subset(scene, torch.tensor(keep, device=scene.means.device))
    _, au = fu(sub, False)
    _, ap = fp(sub, False)
    blended_u, blended_p = bool(au[y, x] > 0), bool(ap[y, x] > 0)
    print(f"\n  worst pixel (y={y}, x={x}): kept set blended in unpacked render: {blended_u}, in packed render: {blended_p}")

    consistent = {}
    if len(keep) == 1:
        g = keep[0]
        for name, px, py in (("center", x + 0.5, y + 0.5), ("corner", float(x), float(y))):
            al = {}
            for lab, d_ in (("unpacked", du), ("packed", dp)):
                al[lab] = alpha_f32(d_["means2d"][g], d_["conics"][g], d_["opacities"][g], px, py)
            pred_u, pred_p = bool(al["unpacked"] >= THRESHOLD), bool(al["packed"] >= THRESHOLD)
            consistent[name] = (pred_u, pred_p) == (blended_u, blended_p)
            print(
                f"  alpha at pixel {name}: unpacked={al['unpacked'].item():.9g} packed={al['packed'].item():.9g} "
                f"(1/255 = {THRESHOLD:.9g}) -> predicts blended unpacked={pred_u}, packed={pred_p}; "
                f"matches the renders: {consistent[name]}"
            )
    stage = "projection outputs differ between the paths" if any_diff else "projection outputs are bit-identical; the difference arises later"
    print(f"  STAGE: {stage}")
    return {"any_diff": any_diff, "consistent": consistent, "n_keep": len(keep)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-idx", type=int, default=None)
    p.add_argument("--scenes-from", type=str, default=None)
    p.add_argument("--where", type=str, default="mr1_verdict=isolated_discontinuity")
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    a.large, a.mode = True, None

    if a.scenes_from:
        col, val = a.where.split("=", 1)
        with open(a.scenes_from) as f:
            idxs = [int(r["scene_idx"]) for r in csv.DictReader(f) if r.get(col) == val][:: a.every]
    elif a.scene_idx is not None:
        idxs = [a.scene_idx]
    else:
        p.error("--scene-idx or --scenes-from is required")

    results = [(i, diagnose(a, i)) for i in idxs]
    print(f"\nSUMMARY of {len(results)} scenes:")
    counts = {"projection differs": 0, "projection identical": 0, "no rgb difference to bisect": 0}
    cons = {"center": [0, 0], "corner": [0, 0]}
    for i, r in results:
        if r is None:
            counts["no rgb difference to bisect"] += 1
            print(f"  scene {i}: nothing to bisect")
            continue
        counts["projection differs" if r["any_diff"] else "projection identical"] += 1
        for k, ok in r["consistent"].items():
            cons[k][0 if ok else 1] += 1
        print(f"  scene {i}: {'projection differs' if r['any_diff'] else 'projection identical'} ({r['n_keep']} kept)")
    print(f"  counts: {counts}")
    print(f"  float32 alpha recomputation matched the renders (single-Gaussian scenes; matched/not): {cons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
