"""Does gsplat's own rasterizer arithmetic explain the blend/skip decisions torch could not reproduce?

diagnose_mr1_inputs.py recomputed a kept Gaussian's alpha in torch from each
path's own float32 conic, and matched the blended/skipped pattern of the two
renders in 42 of 59 MR1 scenes; diagnose_open_items.py's early-stop replay
missed one of two scenes. The rasterizer evaluates

    sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy
    alpha = min(0.999f, opac * __expf(-sigma))
    skip if sigma < 0 or alpha < 1/255;   next_T = T * (1 - alpha); stop if next_T <= 1e-4

with the fast `__expf` intrinsic and nvcc's multiply-add contraction, neither of which
torch reproduces. This script compiles that exact expression with torch's
`load_inline` (same nvcc, -O3 and -use_fast_math as gsplat's build) and checks:

  1  MR1 cutoff flips: does the compiled expression, applied to each path's own conic,
     predict which path blended the Gaussian (59-scene set), against torch float32 and
     against a float64 evaluation?
  2  early stop (two untied scenes): does the compiled front-to-back replay reproduce
     which render stopped?

Usage:
    python gsplat_mt/replicate_kernel_arithmetic.py --csv results/large_scale_w11.csv --csv-xl results/xl_scale_findings.csv
"""

from __future__ import annotations

import argparse
import csv

import torch
from diagnose_mr1 import render_mode
from diagnose_mr1_inputs import dense
from diagnose_mr4 import bisect, load, render, subset
from torch.utils.cpp_extension import load_inline

CPP = "torch::Tensor alpha_sigma(torch::Tensor data); torch::Tensor replay(torch::Tensor data);"

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define ALPHA_THRESHOLD (1.0f / 255.0f)

// data columns: mx, my, conic.x, conic.y, conic.z, opacity, px, py
__global__ void alpha_sigma_kernel(const float *d, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float *r = d + i * 8;
    float dx = r[0] - r[6], dy = r[1] - r[7];
    const float sigma = 0.5f * (r[2] * dx * dx + r[4] * dy * dy) + r[3] * dx * dy;
    float alpha = min(0.999f, r[5] * __expf(-sigma));
    out[i * 2] = alpha;
    out[i * 2 + 1] = sigma;
}

// one thread, Gaussians already sorted front to back: final T and the index where the pixel stopped (-1: none)
__global__ void replay_kernel(const float *d, float *out, int m) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float T = 1.0f;
    float stop = -1.0f;
    for (int i = 0; i < m; ++i) {
        const float *r = d + i * 8;
        float dx = r[0] - r[6], dy = r[1] - r[7];
        const float sigma = 0.5f * (r[2] * dx * dx + r[4] * dy * dy) + r[3] * dx * dy;
        float alpha = min(0.999f, r[5] * __expf(-sigma));
        if (sigma < 0.f || alpha < ALPHA_THRESHOLD) continue;
        const float next_T = T * (1.0f - alpha);
        if (next_T <= 1e-4f) { stop = (float)i; break; }
        T = next_T;
    }
    out[0] = T;
    out[1] = stop;
}

torch::Tensor alpha_sigma(torch::Tensor data) {
    int n = data.size(0);
    auto out = torch::empty({n, 2}, data.options());
    alpha_sigma_kernel<<<(n + 127) / 128, 128>>>(data.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

torch::Tensor replay(torch::Tensor data) {
    auto out = torch::empty({2}, data.options());
    replay_kernel<<<1, 1>>>(data.data_ptr<float>(), out.data_ptr<float>(), data.size(0));
    return out;
}
"""

THR32 = torch.tensor(1.0 / 255.0, dtype=torch.float32)


def build(name, fast):
    flags = ["-O3"] + (["-use_fast_math"] if fast else [])
    return load_inline(
        name=name,
        cpp_sources=CPP,
        cuda_sources=CUDA,
        functions=["alpha_sigma", "replay"],
        extra_cuda_cflags=flags,
        verbose=False,
    )


def rows_for(d, g, x, y, dev):
    """[1, 8] float32 row for Gaussian g at the pixel centre (x + 0.5, y + 0.5)."""
    return torch.cat(
        [d["means2d"][g].float(), d["conics"][g].float(), d["opacities"][g].float().reshape(1), torch.tensor([x + 0.5, y + 0.5], device=dev)]
    ).reshape(1, 8)


def decide_f64(row):
    r = row[0].double()
    dx, dy = r[0] - r[6], r[1] - r[7]
    sigma = 0.5 * (r[2] * dx * dx + r[4] * dy * dy) + r[3] * dx * dy
    alpha = min(0.999, (r[5] * torch.exp(-sigma)).item())
    return bool(alpha >= 1.0 / 255.0 and sigma >= 0)


def part1(a, mods, idxs):
    tot = {k: 0 for k in ("scenes", *mods, "torch32", "f64")}
    for idx in idxs:
        _, scene, cams = load(a, idx)
        cam = cams[0]
        dev = scene.means.device
        fu, fp = render_mode(cam, False), render_mode(cam, True)
        res = bisect(scene, cam, 1.0, by="color", render_a=fu, render_b=fp, k_b=1.0)
        if res is None or len(res["keep"]) != 1:
            continue
        (y, x), g = res["pixel"], res["keep"][0]
        sub = subset(scene, torch.tensor(res["keep"], device=dev))
        blended = (bool(fu(sub, False)[1][y, x] > 0), bool(fp(sub, False)[1][y, x] > 0))
        _, _, mu = fu(scene, True)
        _, _, mp = fp(scene, True)
        du, _ = dense(mu, scene.n, False)
        dp, _ = dense(mp, scene.n, True)
        ru, rp = rows_for(du, g, x, y, dev), rows_for(dp, g, x, y, dev)
        tot["scenes"] += 1
        for name, m in mods.items():
            pu, pp = (m.alpha_sigma(r)[0] for r in (ru, rp))
            pred = tuple(bool(v[0] >= THR32.to(dev) and v[1] >= 0) for v in (pu, pp))
            tot[name] += pred == blended
        # torch float32 with accurate exp, as diagnose_mr1_inputs.py did
        def t32(r):
            r = r[0]
            dx, dy = r[0] - r[6], r[1] - r[7]
            sigma = 0.5 * (r[2] * dx * dx + r[4] * dy * dy) + r[3] * dx * dy
            return bool(torch.clamp(r[5] * torch.exp(-sigma), max=0.999) >= THR32.to(dev))
        tot["torch32"] += (t32(ru), t32(rp)) == blended
        tot["f64"] += (decide_f64(ru), decide_f64(rp)) == blended
    print(f"\nPART 1: {tot['scenes']} single-Gaussian MR1 scenes; predicted (unpacked, packed) blended pattern equals the renders in:")
    for k, v in tot.items():
        if k != "scenes":
            print(f"  {k:28} {v} of {tot['scenes']}")


def part2(a, mod, scenes):
    print("\nPART 2: early-stop replay with the compiled arithmetic")
    for idx in scenes:
        _, scene, cams = load(a, idx)
        cam, dev = cams[0], scene.means.device
        res = bisect(scene, cam, 100.0)
        if res is None:
            continue
        (y, x), keep = res["pixel"], res["keep"]
        sub = subset(scene, torch.tensor(keep, device=dev))
        print(f"\n  scene {idx}, pixel (y={y}, x={x}), {len(keep)} Gaussians")
        for label, kk in (("base", 1.0), ("scaled", 100.0)):
            _, alpha, meta = render(sub, cam, kk, with_meta=True)
            d, _ = dense(meta, len(keep), False)
            order = torch.argsort(meta["depths"][0], stable=True)
            data = torch.cat(
                [
                    d["means2d"][order].float(),
                    d["conics"][order].float(),
                    d["opacities"][order].float().reshape(-1, 1),
                    torch.tensor([x + 0.5, y + 0.5], device=dev).expand(len(keep), 2),
                ],
                1,
            ).contiguous()
            T, stop = (v.item() for v in mod.replay(data))
            print(f"    {label}: replay final T={T:.6e}, stopped at sorted position {int(stop)}; kernel final T={1 - alpha[y, x].item():.6e}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="results/large_scale_w11.csv")
    p.add_argument("--csv-xl", default="results/xl_scale_findings.csv")
    p.add_argument("--every-xl", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    a.large, a.mode, a.scene_idx = True, None, None

    mods = {"compiled -use_fast_math": build("gsplat_replica_fast", True), "compiled, accurate math": build("gsplat_replica_plain", False)}

    def mr1_idxs(path, every):
        with open(path) as f:
            return [int(r["scene_idx"]) for r in csv.DictReader(f) if r["mr1_verdict"] == "isolated_discontinuity"][::every]

    idxs = mr1_idxs(a.csv, 1) + mr1_idxs(a.csv_xl, a.every_xl)
    part1(a, mods, idxs)
    part2(a, mods["compiled -use_fast_math"], [53078, 85315])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
