"""Diagnoses for the open items left after the 100,000-scene run.

Scenes are replayed as run_large_scale.py built them (seed 1000); the default
scene indices are those of the 100,000-scene run.

item 1  smooth difference: an anisotropic scene (N = 27, k = 0.01) where one
        Gaussian with alpha 0.31 differs by 3.5e-4 although nothing is blended or
        skipped. Prints the Gaussian's 2D mean, conic and depth in both renders,
        their ULP and relative differences, and a float64 reference projection
        (my own re-implementation of gsplat's EWA projection, so a mismatch
        with BOTH renders would mean the re-implementation differs from gsplat's).
item 2  a violation whose worst pixel is a plain cutoff flip (N = 17): per factor k,
        rgb and alpha D_max and pixels over tolerance, and which verdict condition
        (pixel count, size bound) each exceeds.
item 3  a cutoff flip with 280 pixels over tolerance: removes the kept Gaussian and
        counts the pixels still over tolerance; shows how close that Gaussian's
        alpha is to 1/255 at every over-tolerance pixel.
item 4  early stop (the 20- and 14-Gaussian sets): replays the kernel's front-to-back
        blending at the worst pixel, in float32, from each render's own values and
        prints the transmittance sequence around the 1e-4 stop.
item 5  MR1 scenes whose only difference is in the alpha image: bisects the alpha image.

Usage:
    python gsplat_mt/diagnose_open_items.py --item all
    python gsplat_mt/diagnose_open_items.py --item 3 --scene-idx 65923 --k 100
"""

from __future__ import annotations

import argparse
import csv
import traceback

import torch
from diagnose_mr1 import render_mode
from diagnose_mr1_inputs import alpha_f32, ulps
from diagnose_mr4 import TOL, bisect, load, meta_at, render, subset

THR = 1.0 / 255.0
ISOLATED_PIXEL_FRAC = 5e-4  # metrics.py
ISOLATED_MAX_D_MAX = 4e-3  # metrics.py


def get(a, idx):
    _, scene, cams = load(a, idx)
    return scene, cams[0]


def f32(x, dev):
    return torch.tensor(x, dtype=torch.float32, device=dev)


# ---------------------------------------------------------------- item 1
def project_ref(scene, cam, k, g):
    """float64 EWA projection of Gaussian g with world scale k applied (mirrors gsplat's persp projection)."""
    d = torch.float64
    mean = scene.means[g].double() * k
    s = scene.scales[g].double() * k
    q = scene.quats[g].double()
    w, x, y, z = q / q.norm()
    R = torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
        ]
    )
    sigma = R @ torch.diag(s * s) @ R.T
    view = cam.viewmat.double().clone()
    view[:3, 3] *= k
    Rc, t = view[:3, :3], view[:3, 3]
    X, Y, Z = Rc @ mean + t
    fx, fy, cx, cy = cam.K[0, 0].double(), cam.K[1, 1].double(), cam.K[0, 2].double(), cam.K[1, 2].double()
    W, H = float(cam.width), float(cam.height)
    tfx, tfy = 0.5 * W / fx, 0.5 * H / fy
    lxp, lxn = (W - cx) / fx + 0.3 * tfx, cx / fx + 0.3 * tfx
    lyp, lyn = (H - cy) / fy + 0.3 * tfy, cy / fy + 0.3 * tfy
    tx = Z * torch.minimum(lxp, torch.maximum(-lxn, X / Z))
    ty = Z * torch.minimum(lyp, torch.maximum(-lyn, Y / Z))
    J = torch.stack(
        [torch.stack([fx / Z, torch.zeros_like(Z), -fx * tx / Z**2]), torch.stack([torch.zeros_like(Z), fy / Z, -fy * ty / Z**2])]
    )
    cov2 = J @ (Rc @ sigma @ Rc.T) @ J.T + 0.3 * torch.eye(2, dtype=d, device=Rc.device)
    det = cov2[0, 0] * cov2[1, 1] - cov2[0, 1] ** 2
    conic = torch.stack([cov2[1, 1] / det, -cov2[0, 1] / det, cov2[0, 0] / det])
    mean2d = torch.stack([fx * X / Z + cx, fy * Y / Z + cy])
    ev = torch.linalg.eigvalsh(cov2)
    return mean2d, conic, Z, (ev[1] / ev[0]).sqrt()


def item1(a, idx, k):
    scene, cam = get(a, idx)
    print(f"scene {idx} n={scene.n} k={k:g}")
    res = bisect(scene, cam, k)
    if res is None:
        return
    (y, x), keep = res["pixel"], res["keep"]
    _, _, m0 = render(scene, cam, 1.0, with_meta=True)
    _, _, mk = render(scene, cam, k, with_meta=True)
    for g in keep[:3]:
        print(f"\n  Gaussian {g}: opacity={scene.opacities[g].item():.6f}")
        ref_m, ref_c, ref_z, axis = project_ref(scene, cam, 1.0, g)
        refk_m, refk_c, refk_z, _ = project_ref(scene, cam, k, g)
        print(f"    screen-space axis ratio (major/minor, from the float64 covariance): {axis.item():.1f}")
        vals = {}
        for key, refb, refs in (("means2d", ref_m, refk_m), ("conics", ref_c, refk_c)):
            b, s = meta_at(m0, key, g), meta_at(mk, key, g)
            vals[key] = (b, s)
            ulp = int(ulps(b, s).max())
            relb = ((b.double() - refb).abs() / refb.abs().clamp_min(1e-30)).max().item()
            rels = ((s.double() - refs).abs() / refs.abs().clamp_min(1e-30)).max().item()
            print(
                f"    {key:8} base={[f'{v:.9g}' for v in b.tolist()]} scaled={[f'{v:.9g}' for v in s.tolist()]}  "
                f"base-vs-scaled: {ulp} ULP | max relative error vs float64 reference: base {relb:.2e}, scaled {rels:.2e}"
            )
        op = scene.opacities[g].double()
        for name, px, py in (("center", x + 0.5, y + 0.5), ("corner", float(x), float(y))):
            ab = alpha_f32(vals["means2d"][0], vals["conics"][0], meta_at(m0, "opacities", g), px, py).item()
            as_ = alpha_f32(vals["means2d"][1], vals["conics"][1], meta_at(mk, "opacities", g), px, py).item()
            dx, dy = ref_m[0] - px, ref_m[1] - py
            ra = min(0.999, (op * torch.exp(-(0.5 * (ref_c[0] * dx * dx + ref_c[2] * dy * dy) + ref_c[1] * dx * dy))).item())
            dxk, dyk = refk_m[0] - px, refk_m[1] - py
            rak = min(0.999, (op * torch.exp(-(0.5 * (refk_c[0] * dxk * dxk + refk_c[2] * dyk * dyk) + refk_c[1] * dxk * dyk))).item())
            print(
                f"    alpha at pixel {name}: base f32={ab:.9g} scaled f32={as_:.9g} | float64 reference: base={ra:.9g} scaled={rak:.9g}"
            )
    print(
        "\n  reading: if one render's conic is far from the float64 reference (relative error well above 1e-6) "
        "and the other is not, projection precision at that scale explains it; if both match the reference "
        "closely, the difference arises in rasterization (or the reference differs from gsplat's formula)."
    )


# ---------------------------------------------------------------- item 2
def item2(a, idx):
    scene, cam = get(a, idx)
    print(f"scene {idx} n={scene.n}")
    rgb0, a0 = render(scene, cam, 1.0)
    npix = rgb0.shape[0] * rgb0.shape[1]
    limit = int(ISOLATED_PIXEL_FRAC * npix)
    print(f"  isolated rule per image: at most {limit} pixels over {TOL:g} and D_max at most {ISOLATED_MAX_D_MAX:g}")
    for k in (0.01, 0.0078125, 0.25, 4.0, 100.0, 128.0):
        rgb, al = render(scene, cam, k)
        for name, d in (("rgb", (rgb - rgb0).abs().amax(-1)), ("alpha", (al - a0).abs())):
            dm, npx = d.max().item(), int((d > TOL).sum())
            note = ""
            if dm > TOL:
                why = []
                if npx > limit:
                    why.append(f"{npx} pixels > {limit}")
                if dm > ISOLATED_MAX_D_MAX:
                    why.append(f"D_max {dm:.2e} > {ISOLATED_MAX_D_MAX:g}")
                note = "  -> violation: " + ", ".join(why) if why else "  -> would be isolated_discontinuity"
            print(f"  k={k:<10g} {name:5} D_max={dm:.3e}  pixels over {TOL:g}: {npx}{note}")


# ---------------------------------------------------------------- item 3
def item3(a, idx, k):
    scene, cam = get(a, idx)
    print(f"scene {idx} n={scene.n} k={k:g}")
    res = bisect(scene, cam, k)
    if res is None:
        return
    keep = res["keep"]
    rgb0, _ = render(scene, cam, 1.0)
    rgbk, _ = render(scene, cam, k)
    over = (rgbk - rgb0).abs().amax(-1) > TOL
    n0 = int(over.sum())
    mask = torch.ones(scene.n, dtype=torch.bool, device=scene.means.device)
    mask[keep] = False
    sub = subset(scene, mask.nonzero().squeeze(-1))
    r0, _ = render(sub, cam, 1.0)
    rk, _ = render(sub, cam, k)
    d1 = (rk - r0).abs().amax(-1)
    print(
        f"\n  pixels over {TOL:g} in the full scene: {n0}\n"
        f"  with the {len(keep)} kept Gaussian(s) removed: {int((d1 > TOL).sum())} pixels over {TOL:g}, D_max {d1.max().item():.3e}"
    )
    if len(keep) == 1:
        _, _, m0 = render(scene, cam, 1.0, with_meta=True)
        g = keep[0]
        ys, xs = torch.nonzero(over, as_tuple=True)
        for name, ox in (("center", 0.5), ("corner", 0.0)):
            al = alpha_f32(meta_at(m0, "means2d", g), meta_at(m0, "conics", g), meta_at(m0, "opacities", g), xs.float() + ox, ys.float() + ox)
            near = int(((al - THR).abs() / THR < 0.01).sum())
            print(
                f"  alpha of that Gaussian at the {n0} over-tolerance pixels ({name}): min={al.min().item():.6f} "
                f"max={al.max().item():.6f}; within 1% of 1/255: {near} of {n0}"
            )


# ---------------------------------------------------------------- item 4
def emulate(meta, m, x, y, label):
    """Front-to-back blending at pixel (y, x) with gsplat's skip and early-stop rules, in float32."""
    dev = meta["depths"].device
    order = torch.argsort(meta["depths"][0], stable=True).tolist()
    T = f32(1.0, dev)
    rows, stopped = [], None
    for j in order:
        if not bool((meta["radii"][0, j] > 0).all()):
            continue
        al = alpha_f32(meta["means2d"][0, j], meta["conics"][0, j], meta["opacities"][0, j], x + 0.5, y + 0.5)
        if al < THR:
            rows.append((j, meta["depths"][0, j].item(), al.item(), T.item(), None, "skipped (alpha < 1/255)"))
            continue
        nt = T * (1 - al)
        if nt <= 1e-4:
            rows.append((j, meta["depths"][0, j].item(), al.item(), T.item(), nt.item(), "STOP: next_T <= 1e-4, not blended"))
            stopped = j
            break
        rows.append((j, meta["depths"][0, j].item(), al.item(), T.item(), nt.item(), "blended"))
        T = nt
    print(f"  {label}: emulated final T = {T.item():.6e}; stopped at Gaussian {stopped}")
    for r in rows[-6:]:
        nt = "-" if r[4] is None else f"{r[4]:.6e}"
        print(f"    Gaussian {r[0]:3d} depth={r[1]:.7g} alpha={r[2]:.6f} T before={r[3]:.6e} next_T={nt}  {r[5]}")
    return T.item()


def item4(a, idx, k):
    scene, cam = get(a, idx)
    print(f"scene {idx} n={scene.n} k={k:g}")
    res = bisect(scene, cam, k)
    if res is None:
        return
    (y, x), keep = res["pixel"], res["keep"]
    sub = subset(scene, torch.tensor(keep, device=scene.means.device))
    print(f"\n  front-to-back replay of the {len(keep)} kept Gaussians at pixel (y={y}, x={x}), pixel-center convention:")
    for label, kk in (("base", 1.0), ("scaled", k)):
        _, alpha, meta = render(sub, cam, kk, with_meta=True)
        emulate(meta, len(keep), x, y, label)
        print(f"    kernel render: final T = {1 - alpha[y, x].item():.6e}")


# ---------------------------------------------------------------- item 5
def item5(a, idx):
    scene, cam = get(a, idx)
    print(f"scene {idx} n={scene.n}")
    fu, fp = render_mode(cam, False), render_mode(cam, True)
    ru, au = fu(scene, False)
    rp, ap = fp(scene, False)
    print(
        f"  rgb: D_max={(ru - rp).abs().max().item():.3e}, pixels over {TOL:g}: {int(((ru - rp).abs().amax(-1) > TOL).sum())} | "
        f"alpha: D_max={(au - ap).abs().max().item():.3e}, pixels over {TOL:g}: {int(((au - ap).abs() > TOL).sum())}"
    )
    bisect(scene, cam, 1.0, by="alpha", render_a=fu, render_b=fp, k_b=1.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--item", default="all", help="1-5 or all")
    p.add_argument("--scene-idx", type=int, nargs="+", default=None, help="override the default scene(s) of the item")
    p.add_argument("--k", type=float, default=None)
    p.add_argument("--csv", default="results/xl_scale_findings.csv", help="item 5: scenes with an MR1 isolated_discontinuity")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    a.large, a.mode = True, None
    items = [1, 2, 3, 4, 5] if a.item == "all" else [int(a.item)]

    def run(n, fn, *args):
        print(f"\n######## item {n}")
        try:
            fn(a, *args)
        except Exception:  # noqa: BLE001 - keep going so one failure does not hide the other items
            traceback.print_exc()

    for n in items:
        if n == 1:
            for i in a.scene_idx or [65342]:
                run(1, item1, i, a.k or 0.01)
        elif n == 2:
            for i in a.scene_idx or [28499]:
                run(2, item2, i)
        elif n == 3:
            for i in a.scene_idx or [65923]:
                run(3, item3, i, a.k or 100.0)
        elif n == 4:
            for i in a.scene_idx or [53078, 85315]:
                run(4, item4, i, a.k or 100.0)
        elif n == 5:
            if a.scene_idx:
                idxs = a.scene_idx
            else:
                with open(a.csv) as f:
                    idxs = [
                        int(r["scene_idx"])
                        for r in csv.DictReader(f)
                        if r["mr1_verdict"] == "isolated_discontinuity" and r["mr1_rgb_d_max"] and float(r["mr1_rgb_d_max"]) <= TOL
                    ]
                print(f"\nitem 5: {len(idxs)} MR1 scenes in {a.csv} with rgb D_max <= {TOL:g} (alpha-only differences)")
            for i in idxs:
                run(5, item5, i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
