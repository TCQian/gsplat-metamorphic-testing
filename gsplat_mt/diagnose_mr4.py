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

Usage:
    python gsplat_mt/diagnose_mr4.py --scene-idx 68 --mode scale_huge
    python gsplat_mt/diagnose_mr4.py --scene-idx 112 --large --bisect
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


def alpha_estimates(meta, g, y, x):
    """Alpha of Gaussian g at pixel (y, x) from gsplat's reported means2d/conics/opacities."""
    try:
        mx, my = meta["means2d"][0, g]
        a, b, c = meta["conics"][0, g]
        op = meta["opacities"][0, g]
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"
    out = []
    for name, px, py in (("pixel center", x + 0.5, y + 0.5), ("pixel corner", float(x), float(y))):
        dx, dy = mx - px, my - py
        sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
        out.append(f"{name}: {min(0.999, (op * torch.exp(-sigma)).item()):.6f}")
    return ", ".join(out)


def bisect(scene, cam, k, tol=TOL):
    rgb0, _ = render(scene, cam, 1.0)
    rgbk, _ = render(scene, cam, k)
    diff = (rgbk - rgb0).abs().amax(-1)
    d0, flat = diff.flatten().max(0)
    y, x = divmod(int(flat), diff.shape[1])
    print(f"\nbisect at k={k:g}: worst pixel (y={y}, x={x}), diff={d0.item():.3e}")
    if d0.item() <= tol:
        print("  nothing over tolerance; skipping")
        return
    thresh = 0.5 * d0.item()
    device = scene.means.device

    def test(idx):
        if not idx:
            return False
        sub = subset(scene, torch.tensor(idx, device=device))
        a, _ = render(sub, cam, 1.0)
        b, _ = render(sub, cam, k)
        return (b[y, x] - a[y, x]).abs().max().item() >= thresh

    keep, evals = ddmin(list(range(scene.n)), test)
    print(f"  smallest set found: {len(keep)} Gaussian(s) of {scene.n} (kept while diff at the pixel stays >= {thresh:.2e}); {evals} test renders")
    sub = subset(scene, torch.tensor(keep, device=device))
    ra, aa, ma = render(sub, cam, 1.0, with_meta=True)
    rb, ab, mb = render(sub, cam, k, with_meta=True)
    print(
        f"  these alone: diff at pixel = {(rb[y, x] - ra[y, x]).abs().max().item():.3e}; "
        f"1-alpha there: base={1 - aa[y, x].item():.6e} scaled={1 - ab[y, x].item():.6e}"
    )
    zs = {}
    for j, g in enumerate(keep[:6]):
        try:
            z0 = ma["depths"][0, j].item()
            zk = mb["depths"][0, j].item() / k
        except Exception as e:  # noqa: BLE001
            z0 = zk = float("nan")
            print(f"  depths unavailable ({type(e).__name__})")
        zs[g] = (z0, zk)
        print(f"  Gaussian {g}: opacity={scene.opacities[g].item():.4f} max_scale={scene.scales[g].max().item():.4g}")
        print(f"    depth (gsplat): base={z0:.9g}  scaled/k={zk:.9g}")
        print(f"    alpha at pixel, base render:   {alpha_estimates(ma, j, y, x)}")
        print(f"    alpha at pixel, scaled render: {alpha_estimates(mb, j, y, x)}")
    gs = list(zs)
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            (a0, ak), (b0, bk) = zs[gs[i]], zs[gs[j]]
            print(
                f"  pair {gs[i]},{gs[j]}: relative depth gap base={abs(a0 - b0) / max(a0, b0):.2e}  "
                f"first in base: {gs[i] if a0 < b0 else gs[j] if b0 < a0 else 'tie'}; "
                f"first in scaled: {gs[i] if ak < bk else gs[j] if bk < ak else 'tie'}"
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-idx", type=int, required=True)
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
    p.add_argument("--bisect", action="store_true", help="find a smallest Gaussian set reproducing the worst pixel")
    p.add_argument("--bisect-k", type=float, default=0.01)
    a = p.parse_args()

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
            world_scales=[0.01, 0.25, 4.0, 100.0],
        )
        cfg = sample_config(a.scene_idx, la)
        mode, args = cfg["mode"], scene_args(cfg, la)
        print(f"config: {cfg}")
    else:
        if a.mode is None:
            p.error("--mode is required without --large")
        mode, args = a.mode, build_argparser().parse_args([])
        args.seed = 0 if a.seed is None else a.seed
    scene, cameras = build_scene(a.scene_idx, mode, args, "cuda")
    cam = cameras[0]
    print(f"scene {a.scene_idx} [{mode}] n={scene.n} res={cam.width}x{cam.height} cameras={len(cameras)}")

    if a.bisect:
        bisect(scene, cam, a.bisect_k)
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
