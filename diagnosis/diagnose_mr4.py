"""Pixel-level diagnosis of an MR4 (world-scale) difference on one scene.

Rebuilds scene `--scene-idx` exactly as runners/run_preliminary.py did (or, with
--large, as runners/run_large_scale.py did; by default the all-camera run, seed
2000 and all nine scene types), then for each factor k renders the original and
the k-scaled scene and reports which pixels differ. Power-of-two
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
mr4_worst_k, at the camera that shows it, falling back to the alpha image when
the rgb image has nothing over tolerance. Each result gets a PATTERN (cutoff
flip, depth-order flip, early stop, smooth difference, other; see pattern_of).
`--out FILE` writes one JSON line per scene as it goes, `--resume` continues a
file after a timeout, and `--summarize FILE...` prints the pattern counts.

The relations render every camera, so a finding can be in any camera's image.
`--camera worst` (the default) picks, per scene, the camera with the largest
difference above the 4e-3 cutoff bound, else the one with the most pixels over
tolerance; `--camera N` fixes the camera. For the earlier camera-0 runs pass
`--seed 1000 --modes` with the original eight scene types and `--camera 0`.

Usage (from the repository root):
    python diagnosis/diagnose_mr4.py --scene-idx 68 --mode scale_huge
    python diagnosis/diagnose_mr4.py --scene-idx 4896 --large --bisect
    python diagnosis/diagnose_mr4.py --scene-idx 4896 --large --bisect --by T
    python diagnosis/diagnose_mr4.py --scenes-from results/allcam_100000_findings.csv --where mr4_verdict=isolated_discontinuity --every 40
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import gsplat
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root, for `import gsplat_mt`

from gsplat_mt.harness import build_argparser, build_scene_and_cameras
from gsplat_mt.metrics import ISOLATED_MAX_D_MAX
from gsplat_mt.sampling import ALLCAM_MODES, ALLCAM_SEED, replay_scene, run_namespace
from gsplat_mt.scene import SCENE_MODES, GaussianScene

NEAR, FAR = 0.01, 100.0
TOL = 1e-4
TILE = 16


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


def render_mode(cam, packed):
    """MR1 render of one camera with packed=True or False: f(scene, with_meta) -> (rgb, alpha[, meta])."""

    def f(scene, with_meta):
        rgb, alpha, meta = gsplat.rasterization(
            means=scene.means,
            quats=scene.quats,
            scales=scene.scales,
            opacities=scene.opacities,
            colors=scene.colors,
            viewmats=cam.viewmat[None],
            Ks=cam.K[None],
            width=cam.width,
            height=cam.height,
            near_plane=NEAR,
            far_plane=FAR,
            sh_degree=None,
            packed=packed,
        )
        if with_meta:
            return rgb[0], alpha[0, ..., 0], meta
        return rgb[0], alpha[0, ..., 0]

    return f


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


def alpha_center(meta, g, y, x):
    """Alpha of Gaussian g at the centre of pixel (y, x), recomputed in torch from gsplat's meta; None if unavailable."""
    try:
        mx, my = meta_at(meta, "means2d", g)
        a, b, c = meta_at(meta, "conics", g)
        op = meta_at(meta, "opacities", g)
    except Exception:  # noqa: BLE001
        return None
    dx, dy = mx - (x + 0.5), my - (y + 0.5)
    sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
    return min(0.999, (op * torch.exp(-sigma)).item())


def condition_numbers(scene, keep, cam):
    """Condition number (larger over smaller eigenvalue) of each kept Gaussian's 2D conic in `cam`."""
    sub = subset(scene, torch.tensor(keep, device=scene.means.device))
    _, _, meta = render(sub, cam, 1.0, with_meta=True)
    out = []
    for a, b, c in meta["conics"][0].tolist():
        tr, det = a + c, a * c - b * b
        disc = max((tr / 2) ** 2 - det, 0.0) ** 0.5
        lo = tr / 2 - disc
        out.append((tr / 2 + disc) / lo if lo > 0 else float("nan"))
    return out


ALPHA_CUTOFF = 1.0 / 255.0


def pattern_of(cls, n_keep, min_t, near_cutoff):
    """The pattern of a bisection result, read off measured values:
    depth-order flip  same transmittance, a kept pair one float32 step apart in depth
    early stop        transmittance changed and final T is near the 1e-4 early-stop threshold
    cutoff flip       transmittance changed and exactly one kept Gaussian has alpha within 1% of 1/255
    smooth difference one kept Gaussian, nothing skipped or reordered
    other             anything else (needs a manual look)
    """
    if cls == "depth-order":
        return "depth-order flip"
    if cls == "transmittance-changed":
        if 1e-5 <= min_t <= 5e-4:
            return "early stop"
        return "cutoff flip" if near_cutoff == 1 else "other"
    if cls == "unclassified" and n_keep == 1:
        return "smooth difference"
    return "other"


def rel_t_change(alpha_a, alpha_b):
    """Relative change in final transmittance T = 1 - alpha between two renders."""
    ta, tb = 1 - alpha_a, 1 - alpha_b
    return (ta - tb).abs() / torch.maximum(ta, tb).clamp_min(1e-12)


def pick_pixel(diff, alpha0, alphak, tol, by):
    """Worst pixel by color difference, or by relative transmittance change among pixels over tol."""
    if by == "color":
        _, flat = diff.flatten().max(0)
    elif by == "alpha":
        _, flat = (alphak - alpha0).abs().flatten().max(0)
    else:
        rel = rel_t_change(alpha0, alphak)
        usable = (diff > tol) & ((1 - alpha0) > 1e-4) & ((1 - alphak) > 1e-4)
        if not usable.any() or rel[usable].max() == 0:
            return None  # no pixel over tol changed its final transmittance
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
    top = (alphak - alpha0).abs().max().item() if by == "alpha" else diff.max().item()
    if pixel is None or top <= tol:
        why = "no over-tolerance pixel changed its final transmittance" if by == "T" and top > tol else "nothing over tolerance"
        print(f"\nbisect at k={k:g}: {why}; skipping")
        return None
    y, x = pixel
    measure_t = by == "T"

    def measure(ra, aa, rb, ab):
        if measure_t:
            return rel_t_change(aa[y, x], ab[y, x]).item()
        if by == "alpha":
            return (ab[y, x] - aa[y, x]).abs().item()
        return (rb[y, x] - ra[y, x]).abs().max().item()

    m0 = measure(rgb0, alpha0, rgbk, alphak)
    print(
        f"\nbisect at k={k:g}: pixel (y={y}, x={x}) chosen by {'transmittance change' if measure_t else by}: "
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
        f"(kept while {'relative T change' if measure_t else 'alpha diff' if by == 'alpha' else 'diff'} at the pixel stays >= {thresh:.2e}); {evals} test renders"
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
    near_cutoff = 0
    for j in range(len(keep)):
        al = [alpha_center(m, j, y, x) for m in (ma, mb)]
        if any(v is not None and abs(v - ALPHA_CUTOFF) <= 0.01 * ALPHA_CUTOFF for v in al):
            near_cutoff += 1
    kappa = condition_numbers(scene, keep, cam) if len(keep) <= 50 else []
    pattern = pattern_of(cls, len(keep), min_t, near_cutoff)
    print(f"  CLASS: {cls} ({len(keep)} Gaussians, rel T change {rel_t:.1e}, diff {(rb[y, x] - ra[y, x]).abs().max().item():.1e}): {note}")
    print(f"  PATTERN: {pattern} ({near_cutoff} kept Gaussian(s) with alpha within 1% of 1/255; kappa {['%.3g' % v for v in kappa[:6]]})")
    return {
        "class": cls,
        "pattern": pattern,
        "n_keep": len(keep),
        "rel_t": rel_t,
        "min_t": min_t,
        "min_gap": min(gaps) if gaps else None,
        "near_cutoff": near_cutoff,
        "kappa": kappa,
        "diff": (rb[y, x] - ra[y, x]).abs().max().item(),
        "keep": keep,
        "pixel": (y, x),
    }


def load(a, scene_idx):
    """(mode, scene, cameras) of one scene.

    With `a.large`: scene `scene_idx` of the large-scale run with seed `a.seed` and scene types
    `a.modes` (defaults: the all-camera run). Otherwise: a preliminary-study scene of type `a.mode`.
    """
    if a.large:
        seed = ALLCAM_SEED if getattr(a, "seed", None) is None else a.seed
        cfg, _, scene, cameras = replay_scene(scene_idx, run_namespace(seed, getattr(a, "modes", None) or ALLCAM_MODES))
        print(f"config: {cfg}")
        return cfg["mode"], scene, cameras
    if a.mode is None:
        raise SystemExit("--mode is required without --large")
    args = build_argparser().parse_args([])
    args.seed = 0 if a.seed is None else a.seed
    scene, cameras = build_scene_and_cameras(scene_idx, a.mode, args, "cuda")
    return a.mode, scene, cameras


def camera_table(cameras, render_a, render_b):
    """Per camera: D_max and pixels over tolerance of the rgb and alpha images between two renders
    (`render_a(cam)` and `render_b(cam)` return (rgb, alpha))."""
    rows = []
    for c, cam in enumerate(cameras):
        rgb_a, alpha_a = render_a(cam)
        rgb_b, alpha_b = render_b(cam)
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


def pick_camera(table, by="color"):
    """The camera whose image most plausibly made the verdict: the largest D_max if it is above the
    cutoff bound, else the most pixels over tolerance (of the alpha image with by="alpha")."""
    dk, pk = ("alpha_d_max", "alpha_px_over") if by == "alpha" else ("d_max", "px_over")
    top = max(table, key=lambda r: r[dk])
    if top[dk] > ISOLATED_MAX_D_MAX:
        return top["camera"]
    return max(table, key=lambda r: (r[pk], r[dk]))["camera"]


def choose_camera(choice, cameras, render_a, render_b, by="color"):
    """Index of the camera to diagnose: `choice` is "worst" or a camera index."""
    if choice != "worst":
        return min(int(choice), len(cameras) - 1)
    if len(cameras) == 1:
        return 0
    table = camera_table(cameras, render_a, render_b)
    c = pick_camera(table, by)
    print("  per camera: " + ", ".join(
        f"{t['camera']}: D_max {t['d_max']:.1e} ({t['px_over']} px), alpha {t['alpha_d_max']:.1e} ({t['alpha_px_over']} px)"
        for t in table))
    return c


def bisect_finding(relation, scene, cameras, k, camera="worst", by="color"):
    """Bisect one MR1 or MR4 finding at the camera that shows it.

    Tries `by` on the rgb image first and falls back to the alpha image, where some findings are the
    only difference (for MR4 also at the other inexact factor). Returns (camera, by, k, result or None).
    """
    attempts = [(by, k)]
    if by != "alpha":
        attempts += [("alpha", kk) for kk in dict.fromkeys([k, 0.01, 100.0] if relation == "mr4" else [k])]
    c = 0
    for by_, kk in attempts:
        if relation == "mr1":
            ra = lambda cm: render_mode(cm, False)(scene, False)  # noqa: E731
            rb = lambda cm: render_mode(cm, True)(scene, False)  # noqa: E731
        else:
            ra = lambda cm: render(scene, cm, 1.0)  # noqa: E731
            rb = lambda cm, kk=kk: render(scene, cm, kk)  # noqa: E731
        c = choose_camera(camera, cameras, ra, rb, by_)
        cam = cameras[c]
        kw = dict(render_a=render_mode(cam, False), render_b=render_mode(cam, True), k_b=1.0) if relation == "mr1" else {}
        r = bisect(scene, cam, kk, by=by_, **kw)
        if r:
            return c, by_, kk, r
    return c, by, k, None


def run_batch(relation, cases, a):
    """Bisect every (scene_idx, k) of `cases`; one JSON line per scene to a.out (appended, so --resume
    skips scenes already written). Prints the pattern counts of this invocation."""
    out = Path(a.out) if a.out else None
    done = set()
    if out and a.resume and out.exists():
        done = {json.loads(line)["scene_idx"] for line in out.read_text().splitlines() if line.strip()}
        print(f"resuming: {len(done)} scenes already in {out}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    f = open(out, "a") if out else None
    try:
        for idx, k in cases:
            if idx in done:
                continue
            if k is None:
                k = 1.0 if relation == "mr1" else getattr(a, "bisect_k", 0.01)
            mode, scene, cameras = load(a, idx)
            print(f"\nscene {idx} [{mode}] n={scene.n} cameras={len(cameras)}")
            c, by, kk, r = bisect_finding(relation, scene, cameras, k, a.camera, a.by)
            pattern = r["pattern"] if r else "nothing over tolerance"
            counts[pattern] = counts.get(pattern, 0) + 1
            rec = {"relation": relation, "where": a.where, "scene_idx": idx, "mode": mode, "n_gaussians": scene.n,
                   "n_cameras": len(cameras), "camera": c, "by": by, "k": kk, "pattern": pattern}
            if r:
                rec.update({key: r[key] for key in ("class", "n_keep", "rel_t", "min_t", "min_gap", "near_cutoff", "diff")})
                rec.update(kappa=r["kappa"][:6], pixel=list(r["pixel"]))
            if f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
    finally:
        if f:
            f.close()
    print(f"\nSUMMARY of {sum(counts.values())} scenes (where {a.where}): {counts}")


def summarize(paths):
    """Pattern counts per relation and selection over JSONL files written by --out."""
    table = {}
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                key = (r["relation"], r["where"])
                table.setdefault(key, {})
                table[key][r["pattern"]] = table[key].get(r["pattern"], 0) + 1
    for (rel, where), c in sorted(table.items()):
        print(f"{rel} {where}: {sum(c.values())} scenes: {dict(sorted(c.items(), key=lambda kv: -kv[1]))}")


def add_batch_args(p):
    p.add_argument("--out", default=None, help="with --scenes-from, write one JSON line per scene here")
    p.add_argument("--resume", action="store_true", help="with --out, skip scenes already in the file")
    p.add_argument("--summarize", nargs="+", default=None, metavar="JSONL", help="print pattern counts of --out files and exit")


def add_camera_arg(p):
    p.add_argument(
        "--camera",
        default="worst",
        help="camera to diagnose: 'worst' (default, the one that shows the finding) or an index",
    )


def cases_from_csv(path, where, rgb_below=None):
    """[(scene_idx, mr4_worst_k)] for rows with column == value, e.g. mr4_verdict=isolated_discontinuity.

    With rgb_below, only rows whose mr4_rgb_d_max is at most that (differences visible only in the alpha image).
    """
    col, val = where.split("=", 1)
    with open(path) as f:
        return [
            (int(r["scene_idx"]), float(r["mr4_worst_k"]) if r.get("mr4_worst_k") else None)
            for r in csv.DictReader(f)
            if r.get(col) == val and (rgb_below is None or (r.get("mr4_rgb_d_max") and float(r["mr4_rgb_d_max"]) <= rgb_below))
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
    p.add_argument("--every", type=int, default=1, help="with --scenes-from, bisect every K-th matching scene")
    p.add_argument("--offset", type=int, default=0, help="with --every, start at the OFFSET-th matching scene (runs of the same K and offsets 0..K-1 together cover all scenes)")
    p.add_argument("--rgb-below", type=float, default=None, help="with --scenes-from, only scenes whose mr4_rgb_d_max is at most this")
    p.add_argument("--mode", type=str, default=None, help="scene type (not used with --large)")
    p.add_argument("--large", action="store_true", help="replay a runners/run_large_scale.py scene")
    p.add_argument("--seed", type=int, default=None, help="default 0, or 2000 (the all-camera run) with --large")
    p.add_argument("--modes", nargs="+", default=None, choices=SCENE_MODES, help="with --large: scene types of the run (default all nine)")
    add_camera_arg(p)
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
        choices=["color", "T", "alpha"],
        default="color",
        help="pixel to bisect: worst color difference, or largest relative change in final transmittance",
    )
    add_batch_args(p)
    a = p.parse_args()

    if a.summarize:
        summarize(a.summarize)
        return 0
    if a.scenes_from:
        a.large = True
        run_batch("mr4", cases_from_csv(a.scenes_from, a.where, a.rgb_below)[a.offset :: a.every], a)
        return 0

    if a.scene_idx is None:
        p.error("--scene-idx or --scenes-from is required")
    mode, scene, cameras = load(a, a.scene_idx)
    c = choose_camera(a.camera, cameras, lambda cm: render(scene, cm, 1.0), lambda cm: render(scene, cm, a.bisect_k))
    cam = cameras[c]
    print(f"scene {a.scene_idx} [{mode}] n={scene.n} res={cam.width}x{cam.height} cameras={len(cameras)} camera={c}")

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
