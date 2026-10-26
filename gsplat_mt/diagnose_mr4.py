"""Pixel-level diagnosis of an MR4 (world-scale) difference on one scene.

Rebuilds scene `--scene-idx` exactly as run_preliminary.py did (or, with
--large, as run_large_scale.py did), then for each factor k renders the
original and the k-scaled scene and reports which pixels differ. Power-of-two
k scale exactly in binary, so a difference that vanishes there is rounding;
one that persists is not.

For pixels over tolerance it also reports how many 16x16 tiles they fall in,
and the alpha of the single Gaussian that would explain the difference. If one
Gaussian is blended in one render and skipped in the other, final transmittance
differs by exactly (1 - alpha) at that pixel, so alpha = 1 - T_small / T_large.
Values at 1/255 (about 0.0039) point to gsplat's alpha cutoff; larger values
point to something else.

Two mechanisms are told apart: a Gaussian blended in one render and skipped in
the other changes final transmittance, whereas two near-equal-depth Gaussians
swapping blend order leave transmittance unchanged but change the color. For
the worst pixels it also lists the Gaussians that could cover the pixel (screen
radius bounded by 3 x largest scale) and the smallest relative depth gap
between depth-adjacent ones; a gap near float rounding (~1e-7) supports a swap.

With --bisect it finds, for the worst pixel at one k, a smallest set of
Gaussians whose removal-by-chunks (delta debugging) still leaves the
base-vs-scaled difference at that pixel, then prints those Gaussians' depths
and alphas as reported by gsplat (meta["depths"], meta["conics"], ...) in both
renders, so the cause can be read off instead of inferred.

Each bisection ends with a `CLASS:` line read off measured values: if the
pixel's final transmittance changed, a Gaussian was blended in one render and
skipped in the other (the line notes when T is near the 1e-4 early-stop
threshold); if it did not change and the kept pair has a depth gap of about one
float32 step, the blend order flipped. `--by T` picks the pixel with the largest
relative transmittance change instead of the largest color difference.
`--scenes-from CSV --where col=value` bisects every matching scene at its
mr4_worst_k and prints a count per class.

Usage:
    python gsplat_mt/diagnose_mr4.py --scene-idx 68 --mode scale_huge
    python gsplat_mt/diagnose_mr4.py --scene-idx 112 --large --bisect
    python gsplat_mt/diagnose_mr4.py --scene-idx 112 --large --bisect --by T
    python gsplat_mt/diagnose_mr4.py --scenes-from results/large_scale.csv --where mr4_verdict=isolated_discontinuity
    python gsplat_mt/diagnose_mr4.py --scene-idx 112 --large
"""

from __future__ import annotations

import argparse

import gsplat
import torch
from run_preliminary import build_argparser
from scene import GaussianScene, sample_cameras, sample_scene

NEAR, FAR = 0.01, 100.0
TOL = 1e-4
TILE = 16


def build_scene(scene_idx, mode, args, device):
    """Mirrors the scene/camera construction in run_preliminary.run_one_scene."""
    generator = torch.Generator(device="cpu").manual_seed(args.seed + scene_idx)
    n = torch.randint(
        args.n_gaussians_min, args.n_gaussians_max + 1, (1,), generator=generator
    ).item()
    scene = sample_scene(
        mode,
        n,
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
    return scene, cameras


def near_ties(scene, cam, y, x):
    """Smallest relative depth gaps among Gaussians that could cover pixel (y, x)."""
    R, t = cam.viewmat[:3, :3], cam.viewmat[:3, 3]
    pts = scene.means @ R.T + t
    z = pts[:, 2]
    f = cam.K[0, 0]
    u = cam.K[0, 0] * pts[:, 0] / z + cam.K[0, 2]
    v = cam.K[1, 1] * pts[:, 1] / z + cam.K[1, 2]
    radius = 3.0 * scene.scales.amax(-1) * f / z
    ok = (z > NEAR) & ((u - x - 0.5).abs() < radius) & ((v - y - 0.5).abs() < radius)
    zc = z[ok].sort().values
    if zc.numel() < 2:
        return zc.numel(), None
    gaps = ((zc[1:] - zc[:-1]) / zc[:-1]).sort().values
    return zc.numel(), gaps[:3].tolist()


def analyze(diff, alpha0, alpha, scene, cam, tol=TOL):
    ys, xs = torch.nonzero(diff > tol, as_tuple=True)
    n = ys.numel()
    if n == 0:
        return
    tiles = len(set(zip((ys // TILE).tolist(), (xs // TILE).tolist())))
    print(f"  pixels over {tol:g}: {n} in {tiles} distinct {TILE}x{TILE} tiles")
    t0, t1 = 1 - alpha0[ys, xs], 1 - alpha[ys, xs]
    ok = (t0 > 2e-3) & (t1 > 2e-3)  # alpha is 1 - T in float32; T this small loses precision
    implied = (1 - torch.minimum(t0, t1) / torch.maximum(t0, t1))[ok]
    print(f"  usable for alpha inference (T > 2e-3): {int(ok.sum())} of {n}")
    if implied.numel():
        near = ((implied > 0.0036) & (implied < 0.0042)).sum().item()
        hi = (implied >= 0.0042).sum().item()
        lo = (implied <= 0.0036).sum().item()
        print(
            f"  implied alpha of the flipped Gaussian: min={implied.min().item():.5f} "
            f"median={implied.median().item():.5f} max={implied.max().item():.5f}  "
            f"| near 1/255 (0.0036-0.0042): {near}, above: {hi}, below: {lo}"
        )
    # Did any Gaussian get added/removed (transmittance changes) or only reordered?
    # float32 alpha resolves a ~0.4% change in T down to about T = 1e-4 (error ~6e-4 relative)
    ok_t = (t0 > 1e-4) & (t1 > 1e-4)
    rel_t = ((t0 - t1).abs() / torch.maximum(t0, t1))[ok_t]
    if rel_t.numel():
        print(
            f"  relative change in final transmittance ({rel_t.numel()} pixels with T > 1e-4): "
            f"median={rel_t.median().item():.2e} max={rel_t.max().item():.2e}  "
            f"(one Gaussian with alpha ~1/255 added or removed would give ~3.9e-03)"
        )
    w = diff.shape[1]
    order = diff[ys, xs].argsort(descending=True)[:3]
    for j in order.tolist():
        y, x = int(ys[j]), int(xs[j])
        cnt, gaps = near_ties(scene, cam, y, x)
        print(
            f"  pixel (y={y}, x={x}): diff={diff[y, x].item():.2e}, {cnt} Gaussians could cover it, "
            f"smallest relative depth gaps: {['%.1e' % g for g in gaps] if gaps else None}"
        )


def render(scene, cam, k, with_meta=False):
    viewmat = cam.viewmat.clone()
    viewmat[:3, 3] *= k
    rgb, alpha, meta = gsplat.rasterization(
        means=scene.means * k,
        quats=scene.quats,
        scales=scene.scales * k,
        opacities=scene.opacities,
        colors=scene.colors,
        viewmats=viewmat[None],
        Ks=cam.K[None],
        width=cam.width,
        height=cam.height,
        near_plane=NEAR * k,
        far_plane=FAR * k,
        sh_degree=None,
        packed=False,
    )
    if with_meta:
        return rgb[0], alpha[0, ..., 0], meta
    return rgb[0], alpha[0, ..., 0]


def subset(scene, idx):
    return GaussianScene(**{f: getattr(scene, f)[idx] for f in ("means", "scales", "quats", "opacities", "colors")})


def ddmin(items, test, max_evals=20000):
    """Zeller's delta debugging: a (locally) 1-minimal subset of `items` for which test() holds."""
    n, evals = 2, 0
    while len(items) >= 2:
        size = -(-len(items) // n)
        chunks = [items[i : i + size] for i in range(0, len(items), size)]
        reduced = False
        for c in chunks:
            evals += 1
            if test(c):
                items, n, reduced = c, 2, True
                break
        if not reduced and len(chunks) > 2:
            for i in range(len(chunks)):
                comp = [g for j, c in enumerate(chunks) if j != i for g in c]
                evals += 1
                if test(comp):
                    items, n, reduced = comp, max(n - 1, 2), True
                    break
        if not reduced:
            if n >= len(items):
                break
            n = min(len(items), n * 2)
        if evals > max_evals:
            print(f"  (stopped after {evals} evaluations)")
            break
    return items, evals


def meta_at(meta, key, g):
    """Entry of a gsplat meta array for (subset-local) Gaussian g, unpacked [C,N,...] or packed [nnz,...]."""
    v = meta[key]
    ids = meta.get("gaussian_ids")
    if isinstance(ids, torch.Tensor) and ids.dim() == 1 and v.shape[0] == ids.shape[0]:
        return v[(ids == g).nonzero()[0, 0]]
    return v[0, g]


def alpha_estimates(meta, g, y, x):
    """Alpha of Gaussian g at pixel (y, x) from gsplat's reported means2d/conics/opacities."""
    try:
        mx, my = meta_at(meta, "means2d", g)
        a, b, c = meta_at(meta, "conics", g)
        op = meta_at(meta, "opacities", g)
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"
    out = []
    for name, px, py in (("pixel center", x + 0.5, y + 0.5), ("pixel corner", float(x), float(y))):
        dx, dy = mx - px, my - py
        sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
        out.append(f"{name}: {min(0.999, (op * torch.exp(-sigma)).item()):.6f}")
    return ", ".join(out)


def rel_t_change(alpha_a, alpha_b):
    """Relative change in final transmittance T = 1 - alpha between two renders."""
    ta, tb = 1 - alpha_a, 1 - alpha_b
    return (ta - tb).abs() / torch.maximum(ta, tb).clamp_min(1e-12)


def pick_pixel(diff, alpha0, alphak, tol, by):
    """Worst pixel by color difference, or by relative transmittance change among pixels over tol."""
    if by == "color":
        _, flat = diff.flatten().max(0)
    else:
        rel = rel_t_change(alpha0, alphak)
        usable = (diff > tol) & ((1 - alpha0) > 1e-4) & ((1 - alphak) > 1e-4)
        if not usable.any():
            return None
        _, flat = torch.where(usable, rel, torch.zeros_like(rel)).flatten().max(0)
    return divmod(int(flat), diff.shape[1])


def bisect(scene, cam, k, tol=TOL, by="color", pixel=None, render_a=None, render_b=None, k_b=None):
    """Delta-debug the Gaussians behind one pixel's base-vs-scaled difference; returns a summary dict.

    The pixel is the worst by color (`by="color"`), the largest relative change in
    final transmittance among pixels over tol (`by="T"`), or `pixel=(y, x)`. The
    measure kept above half its initial value is the same one used to pick it.

    By default the two renders are the original and the k-scaled scene. Pass
    render_a / render_b (callables (scene, with_meta) -> render() outputs) and
    k_b (factor to divide render_b's depths by) to compare other pairs.
    """
    render_a = render_a or (lambda sc, meta: render(sc, cam, 1.0, with_meta=meta))
    render_b = render_b or (lambda sc, meta: render(sc, cam, k, with_meta=meta))
    k_b = k if k_b is None else k_b
    rgb0, alpha0 = render_a(scene, False)
    rgbk, alphak = render_b(scene, False)
    diff = (rgbk - rgb0).abs().amax(-1)
    if pixel is None:
        pixel = pick_pixel(diff, alpha0, alphak, tol, by)
    if pixel is None or diff.max().item() <= tol:
        print(f"\nbisect at k={k:g}: nothing over tolerance; skipping")
        return None
    y, x = pixel
    measure_t = by == "T"

    def measure(ra, aa, rb, ab):
        if measure_t:
            return rel_t_change(aa[y, x], ab[y, x]).item()
        return (rb[y, x] - ra[y, x]).abs().max().item()

    m0 = measure(rgb0, alpha0, rgbk, alphak)
    print(
        f"\nbisect at k={k:g}: pixel (y={y}, x={x}) chosen by {'transmittance change' if measure_t else 'color'}: "
        f"color diff={diff[y, x].item():.3e}, relative change in T={rel_t_change(alpha0[y, x], alphak[y, x]).item():.2e}, "
        f"T base={1 - alpha0[y, x].item():.4e} scaled={1 - alphak[y, x].item():.4e}"
    )
    thresh = 0.5 * m0
    device = scene.means.device

    def test(idx):
        if not idx:
            return False
        sub = subset(scene, torch.tensor(idx, device=device))
        ra, aa = render_a(sub, False)
        rb, ab = render_b(sub, False)
        return measure(ra, aa, rb, ab) >= thresh

    keep, evals = ddmin(list(range(scene.n)), test)
    print(
        f"  smallest set found: {len(keep)} Gaussian(s) of {scene.n} "
        f"(kept while {'relative T change' if measure_t else 'diff'} at the pixel stays >= {thresh:.2e}); {evals} test renders"
    )
    sub = subset(scene, torch.tensor(keep, device=device))
    ra, aa, ma = render_a(sub, True)
    rb, ab, mb = render_b(sub, True)
    t_a, t_b = 1 - aa[y, x].item(), 1 - ab[y, x].item()
    rel_t = rel_t_change(aa[y, x], ab[y, x]).item()
    print(
        f"  these alone: diff at pixel = {(rb[y, x] - ra[y, x]).abs().max().item():.3e}; "
        f"final T there: base={t_a:.6e} scaled={t_b:.6e} (relative change {rel_t:.2e})"
    )
    zs = {}
    for j, g in enumerate(keep[:6]):
        try:
            z0 = meta_at(ma, "depths", j).item()
            zk = meta_at(mb, "depths", j).item() / k_b
        except Exception as e:  # noqa: BLE001
            z0 = zk = float("nan")
            print(f"  depths unavailable ({type(e).__name__})")
        zs[g] = (z0, zk)
        print(f"  Gaussian {g}: opacity={scene.opacities[g].item():.4f} max_scale={scene.scales[g].max().item():.4g}")
        print(f"    depth (gsplat): base={z0:.9g}  scaled/k={zk:.9g}")
        print(f"    alpha at pixel, base render:   {alpha_estimates(ma, j, y, x)}")
        print(f"    alpha at pixel, scaled render: {alpha_estimates(mb, j, y, x)}")
    gs = list(zs)
    gaps = []
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            (a0, ak), (b0, bk) = zs[gs[i]], zs[gs[j]]
            gap = abs(a0 - b0) / max(a0, b0)
            gaps.append(gap)
            print(
                f"  pair {gs[i]},{gs[j]}: relative depth gap base={gap:.2e}  "
                f"first in base: {gs[i] if a0 < b0 else gs[j] if b0 < a0 else 'tie'}; "
                f"first in scaled: {gs[i] if ak < bk else gs[j] if bk < ak else 'tie'}"
            )
    # Read off from measured values only: did T change, or only the order?
    min_t = min(t_a, t_b)
    if rel_t >= 1e-3:
        note = "a Gaussian was blended in one render and skipped in the other"
        if 1e-5 <= min_t <= 5e-4:
            note += f"; final T {min_t:.2e} is near gsplat's early-stop threshold (1e-4)"
        cls = "transmittance-changed"
    elif gaps and min(gaps) <= 3e-7:
        note = f"same transmittance, depth gap {min(gaps):.1e} (about one float32 step)"
        cls = "depth-order"
    else:
        note = "neither a transmittance change nor a near-tied pair"
        cls = "unclassified"
    print(f"  CLASS: {cls} ({len(keep)} Gaussians, rel T change {rel_t:.1e}, diff {(rb[y, x] - ra[y, x]).abs().max().item():.1e}): {note}")
    return {"class": cls, "n_keep": len(keep), "rel_t": rel_t, "min_gap": min(gaps) if gaps else None}


def load(a, scene_idx):
    """(mode, scene, cameras) for one scene, via run_preliminary or run_large_scale defaults."""
    if a.large:
        # Defaults must match run_large_scale.py's argparser (seed, n range, cameras, modes).
        from run_large_scale import sample_config, scene_args
        from scene import SCENE_MODES

        la = argparse.Namespace(
            seed=1000 if a.seed is None else a.seed,
            n_min=10,
            n_max=5000,
            max_cameras=8,
            modes=list(SCENE_MODES),
            world_scales=[0.01, 0.0078125, 0.25, 4.0, 100.0, 128.0],
        )
        cfg = sample_config(scene_idx, la)
        mode, args = cfg["mode"], scene_args(cfg, la)
        print(f"config: {cfg}")
    else:
        if a.mode is None:
            raise SystemExit("--mode is required without --large")
        mode, args = a.mode, build_argparser().parse_args([])
        args.seed = 0 if a.seed is None else a.seed
    scene, cameras = build_scene(scene_idx, mode, args, "cuda")
    return mode, scene, cameras


def cases_from_csv(path, where):
    """[(scene_idx, mr4_worst_k)] for rows with column == value, e.g. mr4_verdict=isolated_discontinuity."""
    import csv

    col, val = where.split("=", 1)
    with open(path) as f:
        return [
            (int(r["scene_idx"]), float(r["mr4_worst_k"]) if r.get("mr4_worst_k") else None)
            for r in csv.DictReader(f)
            if r.get(col) == val
        ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-idx", type=int, default=None)
    p.add_argument(
        "--scenes-from",
        type=str,
        default=None,
        help="results csv; with --where, bisect every matching scene at its mr4_worst_k (implies --bisect --large)",
    )
    p.add_argument("--where", type=str, default="mr4_verdict=isolated_discontinuity", help="column=value filter for --scenes-from")
    p.add_argument("--mode", type=str, default=None, help="scene mode (not used with --large)")
    p.add_argument("--large", action="store_true", help="replay a run_large_scale.py scene")
    p.add_argument("--seed", type=int, default=None, help="default 0, or 1000 with --large")
    p.add_argument(
        "--ks",
        type=float,
        nargs="+",
        default=[1.0, 0.01, 0.0078125, 0.015625, 0.25, 4.0, 64.0, 100.0, 128.0],
    )
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--bisect", action="store_true", help="find a smallest Gaussian set reproducing one pixel's difference")
    p.add_argument("--bisect-k", type=float, default=0.01)
    p.add_argument(
        "--by",
        choices=["color", "T"],
        default="color",
        help="pixel to bisect: worst color difference, or largest relative change in final transmittance",
    )
    a = p.parse_args()

    if a.scenes_from:
        a.large = True
        results = []
        for idx, k in cases_from_csv(a.scenes_from, a.where):
            mode, scene, cameras = load(a, idx)
            cam = cameras[0]
            print(f"scene {idx} [{mode}] n={scene.n} res={cam.width}x{cam.height} cameras={len(cameras)}")
            results.append((idx, bisect(scene, cam, k or a.bisect_k, by=a.by)))
        print(f"\nSUMMARY of {len(results)} scenes (where {a.where}):")
        counts = {}
        for idx, r in results:
            c = r["class"] if r else "nothing over tolerance"
            counts[c] = counts.get(c, 0) + 1
            print(f"  scene {idx}: {c}" + (f" (kept {r['n_keep']}, rel T change {r['rel_t']:.1e})" if r else ""))
        print(f"  counts: {counts}")
        return 0

    if a.scene_idx is None:
        p.error("--scene-idx or --scenes-from is required")
    mode, scene, cameras = load(a, a.scene_idx)
    cam = cameras[0]
    print(f"scene {a.scene_idx} [{mode}] n={scene.n} res={cam.width}x{cam.height} cameras={len(cameras)}")

    if a.bisect:
        bisect(scene, cam, a.bisect_k, by=a.by)
        return 0

    rgb0, alpha0 = render(scene, cam, 1.0)
    rgb0b, _ = render(scene, cam, 1.0)
    print(f"repeat render (determinism): D_max={(rgb0 - rgb0b).abs().max().item():.3e}")

    for k in a.ks:
        rgb, alpha = render(scene, cam, k)
        diff = (rgb - rgb0).abs().amax(-1)  # [H,W]
        pow2 = float(torch.log2(torch.tensor(k)).item()).is_integer()
        print(
            f"\nk={k:g} ({'power of 2' if pow2 else 'not exact'}): "
            f"D_max={diff.max().item():.3e}  D_mae={(rgb - rgb0).abs().mean().item():.3e}  "
            f"pixels differing: >0: {(diff > 0).sum().item()}, >1e-6: {(diff > 1e-6).sum().item()}, "
            f">1e-4: {(diff > 1e-4).sum().item()}"
        )
        analyze(diff, alpha0, alpha, scene, cam)
        w = diff.shape[1]
        vals, idx = diff.flatten().topk(a.top)
        for v, i in zip(vals.tolist(), idx.tolist()):
            y, x = divmod(i, w)
            if v == 0:
                break
            print(
                f"  (y={y:2d}, x={x:2d}) diff={v:.3e}  alpha: base={alpha0[y, x].item():.6f} "
                f"scaled={alpha[y, x].item():.6f}  1-alpha base={1 - alpha0[y, x].item():.3e} "
                f"scaled={1 - alpha[y, x].item():.3e}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
