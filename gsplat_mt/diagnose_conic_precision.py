"""Why do gsplat's conics differ between code paths, and by how much?

gsplat computes a Gaussian's screen-space conic as the inverse of a 2x2
covariance (persp_proj -> add_blur -> glm::inverse, in both the unpacked
`ProjectionEWA3DGSFused.cu` and the packed `ProjectionEWA3DGSPacked.cu`; the
source of the two is the same). Inverting a 2x2 matrix is ill-conditioned when
its axes are very unequal: a relative rounding difference of 6e-8 in the input
becomes about kappa x 6e-8 in the conic, with kappa the ratio of the larger to
the smaller eigenvalue (the squared axis ratio). This script measures that.

For every Gaussian visible in the compared renders it computes kappa from a float64
re-implementation of the projection, then reports, per kappa decade, how far the
conics are apart in ULPs for:
  A  packed vs unpacked (same inputs; MR1),
  B  unpacked, with every scale changed by ONE float32 step (what any last-bit
     difference upstream looks like),
  C  unpacked, base vs world-scaled by k (MR4); k = 0.25 is exact in binary,
and the relative error of each float32 conic against the float64 reference.

If A and C are of the same size as B in every kappa bin, a last-bit difference of any
origin, amplified by conditioning, accounts for the discrepancy; it does not by
itself say which operation differs in the two kernels.

Usage:
    python gsplat_mt/diagnose_conic_precision.py --n-scenes 200
    python gsplat_mt/diagnose_conic_precision.py --gaussian 65923:308 --k 100
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch
from diagnose_mr1 import render_mode
from diagnose_mr1_inputs import dense, ulps
from diagnose_mr4 import load, render

BINS = [(1, 1e1), (1e1, 1e2), (1e2, 1e3), (1e3, 1e4), (1e4, 1e5), (1e5, float("inf"))]


def project_ref_all(scene, cam, k):
    """float64 projection of every Gaussian: mean2d [N,2], conic [N,3], kappa [N] (blurred covariance)."""
    d = torch.float64
    means = scene.means.double() * k
    s = scene.scales.double() * k
    q = scene.quats.double()
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape(-1, 3, 3)
    sigma = (R * (s * s)[:, None, :]) @ R.transpose(1, 2)
    view = cam.viewmat.double().clone()
    view[:3, 3] *= k
    Rc, t = view[:3, :3], view[:3, 3]
    p = means @ Rc.T + t
    X, Y, Z = p.unbind(-1)
    fx, fy, cx, cy = (cam.K[0, 0].double(), cam.K[1, 1].double(), cam.K[0, 2].double(), cam.K[1, 2].double())
    W, H = float(cam.width), float(cam.height)
    tfx, tfy = 0.5 * W / fx, 0.5 * H / fy
    lxp, lxn = (W - cx) / fx + 0.3 * tfx, cx / fx + 0.3 * tfx
    lyp, lyn = (H - cy) / fy + 0.3 * tfy, cy / fy + 0.3 * tfy
    tx = Z * torch.minimum(lxp, torch.maximum(-lxn, X / Z))
    ty = Z * torch.minimum(lyp, torch.maximum(-lyn, Y / Z))
    zero = torch.zeros_like(Z)
    J = torch.stack([fx / Z, zero, -fx * tx / Z**2, zero, fy / Z, -fy * ty / Z**2], -1).reshape(-1, 2, 3)
    cov2 = J @ (Rc @ sigma @ Rc.T) @ J.transpose(1, 2)
    cov2 = cov2 + 0.3 * torch.eye(2, dtype=d, device=cov2.device)
    a, b, c = cov2[:, 0, 0], cov2[:, 0, 1], cov2[:, 1, 1]
    det = a * c - b * b
    conic = torch.stack([c / det, -b / det, a / det], -1)
    mean2d = torch.stack([fx * X / Z + cx, fy * Y / Z + cy], -1)
    half = (a + c) / 2
    root = (half * half - det).clamp_min(0).sqrt()
    kappa = (half + root) / (half - root).clamp_min(1e-300)
    return mean2d, conic, kappa


def meta_dense(scene, cam, k, packed):
    fn = render_mode(cam, packed) if k == 1.0 else None
    if fn is not None:
        _, _, meta = fn(scene, True)
    else:
        _, _, meta = render(scene, cam, k, with_meta=True)
    return dense(meta, scene.n, packed)


def perturbed(scene):
    """Same scene with every scale moved up by one float32 step."""
    return replace(scene, scales=torch.nextafter(scene.scales, torch.full_like(scene.scales, float("inf"))))


class Acc:
    """Values collected per kappa decade."""

    def __init__(self, fmt="{:.0f}"):
        self.vals = {i: [] for i in range(len(BINS))}
        self.fmt = fmt

    def add(self, kappa, v):
        for i, (lo, hi) in enumerate(BINS):
            m = (kappa >= lo) & (kappa < hi)
            if m.any():
                self.vals[i].append(v[m].double().cpu())

    def show(self, title):
        print(f"\n{title}")
        print(f"  {'kappa':>13} {'n':>8} {'median':>10} {'p90':>10} {'max':>10}")
        for i, (lo, hi) in enumerate(BINS):
            if not self.vals[i]:
                continue
            u = torch.cat(self.vals[i])
            f = self.fmt.format
            print(f"  {lo:>5g}-{hi:<7g} {u.numel():>8} {f(u.median().item()):>10} {f(u.quantile(0.9).item()):>10} {f(u.max().item()):>10}")


def conic_ulp(a, b):
    return ulps(a, b).amax(-1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-scenes", type=int, default=200, help="scenes 0..n-1 of the large-scale sampler")
    p.add_argument("--k", type=float, default=0.01)
    p.add_argument("--gaussian", type=str, default=None, help="scene:gaussian, report kappa and conic differences of one Gaussian")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    a.large, a.mode, a.scene_idx = True, None, None

    if a.gaussian:
        si, gi = (int(v) for v in a.gaussian.split(":"))
        _, scene, cams = load(a, si)
        cam = cams[0]
        _, c0, kap = project_ref_all(scene, cam, 1.0)
        print(f"\nscene {si} Gaussian {gi}: kappa (float64) = {kap[gi].item():.1f}, axis ratio = {kap[gi].sqrt().item():.1f}")
        du, vu = meta_dense(scene, cam, 1.0, False)
        dk, vk = meta_dense(scene, cam, a.k, False)
        dp, vp = meta_dense(scene, cam, 1.0, True)
        ck = project_ref_all(scene, cam, a.k)[1]
        dq, _ = meta_dense(perturbed(scene), cam, 1.0, False)
        print(f"  base vs the same scene with every scale moved by ONE float32 step: conic {int(conic_ulp(du['conics'][gi:gi+1], dq['conics'][gi:gi+1]))} ULP (compare: base vs scaled(k={a.k:g}) above)")
        print(f"  base-vs-scaled(k={a.k:g}) conic: {int(conic_ulp(du['conics'][gi:gi+1], dk['conics'][gi:gi+1]))} ULP; unpacked-vs-packed: {int(conic_ulp(du['conics'][gi:gi+1], dp['conics'][gi:gi+1]))} ULP")
        print(f"  relative error vs float64: unpacked {((du['conics'][gi].double()-c0[gi]).abs()/c0[gi].abs()).max().item():.2e}, scaled {((dk['conics'][gi].double()-ck[gi]).abs()/ck[gi].abs()).max().item():.2e}")
        return 0

    acc_pu, acc_pert, acc_k, acc_exact = Acc(), Acc(), Acc(), Acc()
    err_u, err_p = Acc("{:.1e}"), Acc("{:.1e}")
    n_g = 0
    for i in range(a.n_scenes):
        _, scene, cams = load(a, i)
        cam = cams[0]
        _, c_ref, kappa = project_ref_all(scene, cam, 1.0)
        du, vu = meta_dense(scene, cam, 1.0, False)
        dp, vp = meta_dense(scene, cam, 1.0, True)
        dq, vq = meta_dense(perturbed(scene), cam, 1.0, False)
        dk, vk = meta_dense(scene, cam, a.k, False)
        dx, vx = meta_dense(scene, cam, 0.25, False)
        m = vu & vp & vq & vk & vx
        if not m.any():
            continue
        n_g += int(m.sum())
        cu = du["conics"][m]
        kap = kappa[m]
        acc_pu.add(kap, conic_ulp(cu, dp["conics"][m]))
        acc_pert.add(kap, conic_ulp(cu, dq["conics"][m]))
        acc_k.add(kap, conic_ulp(cu, dk["conics"][m]))
        acc_exact.add(kap, conic_ulp(cu, dx["conics"][m]))
        rel = lambda c: ((c.double() - c_ref[m]).abs() / c_ref[m].abs()).amax(-1)
        err_u.add(kap, rel(cu))
        err_p.add(kap, rel(dp["conics"][m]))
    print(f"{a.n_scenes} scenes, {n_g} Gaussians visible in all compared renders")
    acc_pu.show("A  packed vs unpacked: conic distance in ULP")
    acc_pert.show("B  unpacked, every scale moved by ONE float32 step: conic distance in ULP")
    acc_k.show(f"C  unpacked, base vs world scale k = {a.k:g}: conic distance in ULP")
    acc_exact.show("C' unpacked, base vs world scale k = 0.25 (exact in binary): conic distance in ULP")
    err_u.show("relative error of the unpacked conic against the float64 reference")
    err_p.show("relative error of the packed conic against the float64 reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
