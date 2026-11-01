"""Which stage of gsplat's projection first differs between the packed and unpacked kernels?

The unpacked (`ProjectionEWA3DGSFused.cu`) and packed (`ProjectionEWA3DGSPacked.cu`)
kernels run the same source for the covariance chain
    quat_scale_to_covar_preci -> covarW2C -> persp_proj -> add_blur -> glm::inverse
yet about half of the conics differ in the last bits.

Recompiling the real source files with stores of EVERY intermediate matrix made both kernels
bit-identical to each other and unfaithful to gsplat's own build (86% / 48% of conics equal the
real ones), so the stores themselves change the compiled arithmetic (a stored intermediate
cannot be fused away). This version perturbs as little as possible: it reads the installed
files, inserts a store of ONE intermediate (or none), builds each kernel as its own translation
unit with gsplat's flags (`-O3 -use_fast_math`, same includes), and for every build checks
that the kernel's own conic output equals gsplat's real conic bit for bit. Only when both
kernels stay faithful are the stored intermediates compared between them.

  stage 0  no store (does a rebuild of the unmodified source reproduce gsplat's conics?)
  stage 1  world covariance (quat_scale_to_covar_preci)
  stage 2  camera-space covariance (covarW2C)
  stage 3  blurred 2D covariance and its determinant (persp_proj + add_blur)

Usage:
    MAX_JOBS=1 python gsplat_mt/isolate_projection_stage.py --n-scenes 100 --stages 0 1 2 3
"""

from __future__ import annotations

import argparse
import os
import tempfile

import gsplat
import numpy as np
import torch
from diagnose_mr1 import render_mode
from diagnose_mr1_inputs import dense
from diagnose_mr4 import load
from torch.utils.cpp_extension import load as load_ext

NDUMP = 32

HELPERS = r"""
static __device__ float *g_dump = nullptr;
#define NDUMP 32
"""

FUSED_LAUNCH = r"""
std::vector<torch::Tensor> dump_fused(torch::Tensor means, torch::Tensor quats, torch::Tensor scales, torch::Tensor opacities,
                         torch::Tensor viewmat, torch::Tensor K, int64_t W, int64_t H, double eps2d,
                         double near_plane, double far_plane, torch::Tensor covars) {
    uint32_t N = means.size(0);
    auto dump = torch::full({(int64_t)N, 32}, std::numeric_limits<float>::quiet_NaN(), means.options());
    float *dptr = dump.data_ptr<float>();
    cudaMemcpyToSymbol(gsplat::g_dump, &dptr, sizeof(float *));
    auto i32 = means.options().dtype(torch::kInt32);
    auto radii = torch::zeros({(int64_t)N, 2}, i32);
    auto means2d = torch::zeros({(int64_t)N, 2}, means.options());
    auto depths = torch::zeros({(int64_t)N}, means.options());
    auto conics = torch::zeros({(int64_t)N, 3}, means.options());
    gsplat::projection_ewa_3dgs_fused_fwd_kernel<float><<<(N + 255) / 256, 256>>>(
        1, 1, N, means.data_ptr<float>(), covars.numel() > 0 ? covars.data_ptr<float>() : nullptr, quats.data_ptr<float>(), scales.data_ptr<float>(),
        opacities.data_ptr<float>(), viewmat.data_ptr<float>(), K.data_ptr<float>(), W, H, (float)eps2d,
        (float)near_plane, (float)far_plane, 0.0f, gsplat::CameraModelType::PINHOLE, radii.data_ptr<int32_t>(),
        means2d.data_ptr<float>(), depths.data_ptr<float>(), conics.data_ptr<float>(), nullptr);
    cudaDeviceSynchronize();
    float *np = nullptr;
    cudaMemcpyToSymbol(gsplat::g_dump, &np, sizeof(float *));
    return {dump, conics};
}
"""

PACKED_LAUNCH = r"""
std::vector<torch::Tensor> dump_packed(torch::Tensor means, torch::Tensor quats, torch::Tensor scales, torch::Tensor opacities,
                          torch::Tensor viewmat, torch::Tensor K, int64_t W, int64_t H, double eps2d,
                          double near_plane, double far_plane, torch::Tensor covars) {
    uint32_t N = means.size(0);
    auto dump = torch::full({(int64_t)N, 32}, std::numeric_limits<float>::quiet_NaN(), means.options());
    float *dptr = dump.data_ptr<float>();
    cudaMemcpyToSymbol(gsplat::g_dump, &dptr, sizeof(float *));
    dim3 threads(N_THREADS_PACKED);
    dim3 grid((N + threads.x - 1) / threads.x, 1, 1);
    auto i32 = means.options().dtype(torch::kInt32);
    auto i64 = means.options().dtype(torch::kInt64);
    auto block_cnts = torch::zeros({(int64_t)grid.x}, i32);
    // pass 1 of the real launcher: counts per block (the dump, if any, is written here)
    gsplat::projection_ewa_3dgs_packed_fwd_kernel<float><<<grid, threads>>>(
        1, 1, N, means.data_ptr<float>(), covars.numel() > 0 ? covars.data_ptr<float>() : nullptr, quats.data_ptr<float>(), scales.data_ptr<float>(),
        opacities.data_ptr<float>(), viewmat.data_ptr<float>(), K.data_ptr<float>(), W, H, (float)eps2d,
        (float)near_plane, (float)far_plane, 0.0f, nullptr, gsplat::CameraModelType::PINHOLE,
        block_cnts.data_ptr<int32_t>(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr);
    cudaDeviceSynchronize();
    float *np = nullptr;
    cudaMemcpyToSymbol(gsplat::g_dump, &np, sizeof(float *));
    // pass 2: write the packed outputs (as the real launcher does), then scatter the conics back by Gaussian id
    auto block_accum = torch::cumsum(block_cnts, 0).to(torch::kInt32).contiguous();
    int64_t nnz = block_accum[-1].item<int64_t>();
    auto indptr = torch::zeros({2}, i32);
    auto batch_ids = torch::zeros({nnz}, i64), camera_ids = torch::zeros({nnz}, i64), gaussian_ids = torch::zeros({nnz}, i64);
    auto radii = torch::zeros({nnz, 2}, i32);
    auto means2d = torch::zeros({nnz, 2}, means.options());
    auto depths = torch::zeros({nnz}, means.options());
    auto pconics = torch::zeros({nnz, 3}, means.options());
    gsplat::projection_ewa_3dgs_packed_fwd_kernel<float><<<grid, threads>>>(
        1, 1, N, means.data_ptr<float>(), covars.numel() > 0 ? covars.data_ptr<float>() : nullptr, quats.data_ptr<float>(), scales.data_ptr<float>(),
        opacities.data_ptr<float>(), viewmat.data_ptr<float>(), K.data_ptr<float>(), W, H, (float)eps2d,
        (float)near_plane, (float)far_plane, 0.0f, block_accum.data_ptr<int32_t>(),
        gsplat::CameraModelType::PINHOLE, nullptr, indptr.data_ptr<int32_t>(), batch_ids.data_ptr<int64_t>(),
        camera_ids.data_ptr<int64_t>(), gaussian_ids.data_ptr<int64_t>(), radii.data_ptr<int32_t>(),
        means2d.data_ptr<float>(), depths.data_ptr<float>(), pconics.data_ptr<float>(), nullptr);
    cudaDeviceSynchronize();
    auto conics = torch::zeros({(int64_t)N, 3}, means.options());
    if (nnz > 0) conics.index_put_({gaussian_ids}, pconics);
    return {dump, conics};
}
"""

BINDING = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> dump_fused(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double, double, double, torch::Tensor);
std::vector<torch::Tensor> dump_packed(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double, double, double, torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dump_fused", &dump_fused);
    m.def("dump_packed", &dump_packed);
}
"""


def store_text(stage):
    pre = "if (gsplat::g_dump != nullptr) { float *d_ = gsplat::g_dump + (size_t)gid * NDUMP; int k_ = 0; "
    if stage == 0:
        return ""
    if stage == 1:
        return pre + "for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) d_[k_++] = covar[i][j]; }"
    if stage == 2:
        return pre + "k_ = 9; for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) d_[k_++] = covar_c[i][j]; }"
    if stage == 3:
        return pre + "k_ = 18; d_[k_++] = covar2d[0][0]; d_[k_++] = covar2d[0][1]; d_[k_++] = covar2d[1][0]; d_[k_++] = covar2d[1][1]; d_[k_++] = det; }"
    if stage == 4:
        return pre + "k_ = 24; d_[k_++] = covar2d[0][0]; d_[k_++] = covar2d[0][1]; d_[k_++] = covar2d[1][0]; d_[k_++] = covar2d[1][1]; k_ = 28; d_[k_++] = mean_c.x; d_[k_++] = mean_c.y; d_[k_++] = mean_c.z; }"
    raise ValueError(stage)


STAGE_COLS = {1: range(0, 9), 2: range(9, 18), 3: range(18, 23), 4: range(24, 28)}
STAGE_NAME = {
    0: "no store (rebuild of the unmodified source)",
    1: "world covariance (quat_scale_to_covar_preci)",
    2: "camera-space covariance (covarW2C)",
    3: "blurred 2D covariance + determinant (persp_proj + add_blur)",
    4: "2D covariance BEFORE the blur (persp_proj output)",
}


def build(stage):
    root = os.path.join(os.path.dirname(gsplat.__file__), "cuda")
    csrc = os.path.join(root, "csrc")
    out_dir = tempfile.mkdtemp(prefix=f"gsplat_dump{stage}_")

    def patch(fname, anchor, launch, blur_anchor):
        text = open(os.path.join(csrc, fname)).read()
        assert text.count("namespace cg = cooperative_groups;") == 1 and text.count(anchor) == 1, fname
        text = text.replace("namespace cg = cooperative_groups;", "namespace cg = cooperative_groups;\n" + HELPERS)
        if stage == 4:
            assert text.count(blur_anchor) == 1, fname
            text = text.replace(blur_anchor, store_text(stage) + "\n" + blur_anchor)
        else:
            text = text.replace(anchor, anchor + "\n" + store_text(stage))
        text += "\n#include <torch/extension.h>\n#include <limits>\n" + launch
        path = os.path.join(out_dir, "dump_" + fname)
        open(path, "w").write(text)
        return path

    f = patch("ProjectionEWA3DGSFused.cu", "mat2 covar2d_inv = glm::inverse(covar2d);", FUSED_LAUNCH, "float det = add_blur(eps2d, covar2d, compensation);")
    pk = patch("ProjectionEWA3DGSPacked.cu", "covar2d_inv = glm::inverse(covar2d);", PACKED_LAUNCH, "det = add_blur(eps2d, covar2d, compensation);")
    b = os.path.join(out_dir, "binding.cpp")
    open(b, "w").write(BINDING)
    return load_ext(
        name=f"gsplat_proj_dump_s{stage}_sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}",
        sources=[f, pk, b],
        extra_include_paths=[csrc, os.path.join(root, "include"), os.path.join(csrc, "third_party", "glm")],
        extra_cflags=["-O3", "-Wno-attributes"],
        extra_cuda_cflags=["-O3", "-use_fast_math"],
        verbose=False,
    )


def bits(x):
    return x.contiguous().view(torch.int32)


def run_stage(a, stage):
    mod = build(stage)
    tot = {"n": 0, "F": 0, "P": 0, "real_diff": 0, "stage_diff": 0, "same_in_conic_diff": 0, "diff_in_conic_same": 0}
    fields = {}
    for i in range(a.n_scenes):
        _, scene, cams = load(a, i)
        cam = cams[0]
        args = (
            scene.means.contiguous(), scene.quats.contiguous(), scene.scales.contiguous(), scene.opacities.contiguous(),
            cam.viewmat.contiguous(), cam.K.contiguous(), cam.width, cam.height, 0.3, 0.01, 100.0, torch.empty(0, device=scene.means.device),
        )
        dF, cF = mod.dump_fused(*args)
        dP, cP = mod.dump_packed(*args)
        _, _, mu = render_mode(cam, False)(scene, True)
        _, _, mp = render_mode(cam, True)(scene, True)
        du, vu = dense(mu, scene.n, False)
        dp, vp = dense(mp, scene.n, True)
        ok = vu & vp
        if not ok.any():
            continue
        tot["n"] += int(ok.sum())
        tot["F"] += int((bits(cF[ok]) == bits(du["conics"][ok])).all(-1).sum())
        tot["P"] += int((bits(cP[ok]) == bits(dp["conics"][ok])).all(-1).sum())
        tot["real_diff"] += int((bits(du["conics"][ok]) != bits(dp["conics"][ok])).any(-1).sum())
        if stage > 0:
            c = list(STAGE_COLS[stage])
            sd = (bits(dF[ok][:, c]) != bits(dP[ok][:, c]))
            tot["stage_diff"] += int(sd.any(-1).sum())
            cd = (bits(cF[ok]) != bits(cP[ok])).any(-1)
            tot["same_in_conic_diff"] += int((~sd.any(-1) & cd).sum())
            tot["diff_in_conic_same"] += int((sd.any(-1) & ~cd).sum())
            for j, col in enumerate(c):
                fields[col] = fields.get(col, 0) + int(sd[:, j].sum())
    n = tot["n"]
    print(f"\n=== stage {stage}: {STAGE_NAME[stage]}   ({a.n_scenes} scenes, {n} Gaussians visible in both real renders)")
    print(f"  real unpacked vs real packed conic differ in        {tot['real_diff']} ({100*tot['real_diff']/n:.1f}%)")
    print(f"  rebuilt fused kernel equals real unpacked conic in   {tot['F']} ({100*tot['F']/n:.1f}%)")
    print(f"  rebuilt packed kernel equals real packed conic in    {tot['P']} ({100*tot['P']/n:.1f}%)")
    if stage > 0:
        print(f"  stored intermediate differs between the two kernels in {tot['stage_diff']} ({100*tot['stage_diff']/n:.1f}%); per stored value: {dict(fields)}")
        print(f"  stored intermediate identical but the kernels' conics still differ: {tot['same_in_conic_diff']}")
        print(f"  stored intermediate differs but the kernels' conics are identical:  {tot['diff_in_conic_same']}")
    return tot


def fma32(a, b, c):
    """Fused multiply-add on float32 arrays: one rounding of a*b + c (a*b is exact in float64)."""
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)


def two_term(a, b, c, d):
    """The three ways a compiler can evaluate a*b + c*d in float32: [unfused, a*b rounded then fma, c*d rounded then fma]."""
    p, q = (a * b).astype(np.float32), (c * d).astype(np.float32)
    return [(p + q).astype(np.float32), fma32(c, d, p), fma32(a, b, q)]


VARIANT = ["unfused", "a*b rounded, c*d fused", "c*d rounded, a*b fused"]


def pairing_experiment(a):
    """Which rounding pattern does each kernel use for the products of covarW2C = (R S) R^T?

    Gaussians sit exactly on the optical axis and the camera rotates about z only, so after covarW2C
    everything in persp_proj is exact multiplications that this script reproduces, and every
    sum in covarW2C has two non-zero terms. The kernels' pre-blur 2D covariances are matched against all
    3^4 rounding patterns per row of R S R^T; the best match names the pattern each kernel used.
    """
    dev = "cuda"
    mod = build(4)
    N, ncam, W, H, fx = 20000, 24, 256, 256, 300.0
    g = torch.Generator().manual_seed(1234)
    z = (1 + 5 * torch.rand(N, generator=g)).float()
    A = torch.randn(N, 3, 3, generator=g) * 0.1
    S = A @ A.transpose(1, 2)
    S = ((S + S.transpose(1, 2)) / 2).float()
    covars6 = torch.stack([S[:, 0, 0], S[:, 0, 1], S[:, 0, 2], S[:, 1, 1], S[:, 1, 2], S[:, 2, 2]], -1).contiguous()
    means = torch.zeros(N, 3)
    means[:, 2] = z
    quats = torch.tensor([[1.0, 0, 0, 0]]).repeat(N, 1)
    scales = torch.ones(N, 3)
    opac = torch.full((N,), 0.9)
    K = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    to = lambda t: t.to(dev).contiguous()
    base = [to(means), to(quats), to(scales), to(opac)]

    def run(theta):
        vm = torch.eye(4)
        if theta is not None:
            c, s_ = torch.cos(theta), torch.sin(theta)
            vm[0, 0], vm[0, 1], vm[1, 0], vm[1, 1] = c, -s_, s_, c
        args = base + [to(vm), to(K), W, H, 0.3, 0.01, 100.0, to(covars6)]
        dF, cF = mod.dump_fused(*args)
        dP, cP = mod.dump_packed(*args)
        return vm.numpy(), dF[:, 24:28].cpu().numpy(), dP[:, 24:28].cpu().numpy(), cF, cP

    # faithfulness of the instrumented kernels on this input (same covariance fed to gsplat's own render)
    theta0 = torch.tensor(1.0, dtype=torch.float32)
    vm0 = torch.eye(4)
    c0, s0 = torch.cos(theta0), torch.sin(theta0)
    vm0[0, 0], vm0[0, 1], vm0[1, 0], vm0[1, 1] = c0, -s0, s0, c0
    _, _, _, cF, cP = run(theta0)
    real = {}
    for packed in (False, True):
        _, _, meta = gsplat.rasterization(
            means=to(means), quats=to(quats), scales=to(scales), opacities=to(opac), colors=to(torch.ones(N, 3)),
            viewmats=to(vm0)[None], Ks=to(K)[None], width=W, height=H, near_plane=0.01, far_plane=100.0,
            sh_degree=None, packed=packed, covars=to(S),
        )
        real[packed] = dense(meta, N, packed)
    for name, mine, packed in (("fused", cF, False), ("packed", cP, True)):
        d, vis = real[packed]
        ok = vis
        print(f"  instrumented {name} kernel equals gsplat's conic on {int((bits(mine[ok]) == bits(d['conics'][ok])).all(-1).sum())} of {int(ok.sum())} Gaussians")

    # calibration with the identity rotation: find the reciprocal rz the kernel used for each Gaussian
    _, oF, oP, _, _ = run(None)
    zc = z.numpy().astype(np.float32)
    S_np = S.numpy().astype(np.float32)
    fxn = np.float32(fx)
    r0 = (np.float32(1.0) / zc).astype(np.float32)
    cands = [np.nextafter(r0, np.float32(0)), r0, np.nextafter(r0, np.float32(np.inf))]
    chosen = np.full(N, -1)
    for ci, rz in enumerate(cands):
        J = (fxn * rz).astype(np.float32)
        p00 = ((J * S_np[:, 0, 0]).astype(np.float32) * J).astype(np.float32)
        p11 = ((J * S_np[:, 1, 1]).astype(np.float32) * J).astype(np.float32)
        p01 = ((J * S_np[:, 0, 1]).astype(np.float32) * J).astype(np.float32)
        okc = (
            (p00.view(np.int32) == oF[:, 0].view(np.int32)) & (p11.view(np.int32) == oF[:, 3].view(np.int32))
            & (p01.view(np.int32) == oF[:, 1].view(np.int32)) & (p01.view(np.int32) == oF[:, 2].view(np.int32))
            & (p00.view(np.int32) == oP[:, 0].view(np.int32)) & (p11.view(np.int32) == oP[:, 3].view(np.int32))
        )
        chosen = np.where((chosen < 0) & okc, ci, chosen)
    usable = chosen >= 0
    print(f"  calibration (identity rotation): the exact multiplications in persp_proj reproduce both kernels' 2D covariance for {int(usable.sum())} of {N} Gaussians (reciprocal candidates: nearest rounded, +-1 ULP; used {np.bincount(chosen[usable], minlength=3).tolist()})")
    if usable.sum() < 0.9 * N:
        print("  the model of persp_proj does not reproduce the kernels; stopping")
        return
    rz = np.choose(np.where(usable, chosen, 1), cands).astype(np.float32)
    J = (fxn * rz).astype(np.float32)

    score = {k: np.zeros((2, 3, 3, 3, 3), dtype=np.int64) for k in ("fused", "packed")}
    n_samples = 0
    for ci in range(ncam):
        theta = torch.tensor(float(torch.rand(1, generator=g)) * 6.2831853, dtype=torch.float32)
        vm, oF, oP, _, _ = run(theta)
        R = vm[:3, :3].astype(np.float32)
        sel = usable
        Ssel = S_np[sel]
        Jsel = J[sel]
        n_samples += int(sel.sum())
        # evaluate row groups with explicit loops over the variants
        for r in (0, 1):
            # (R S)[r][j] = R[r][0]*S[0][j] + R[r][1]*S[1][j]   (R[r][2] = 0)
            Mr = [two_term(R[r, 0], Ssel[:, 0, j], R[r, 1], Ssel[:, 1, j]) for j in (0, 1)]
            for v0 in range(3):
                for v1 in range(3):
                    # sigma_c[r][c] = M[r][0]*R[c][0] + M[r][1]*R[c][1]
                    sc = [two_term(Mr[0][v0], R[c, 0], Mr[1][v1], R[c, 1]) for c in (0, 1)]
                    for w0 in range(3):
                        for w1 in range(3):
                            # elem(r, c) = (J_rr * sigma_c[r][c]) * J_cc
                            e0 = ((Jsel * sc[0][w0]).astype(np.float32) * Jsel).astype(np.float32)
                            e1 = ((Jsel * sc[1][w1]).astype(np.float32) * Jsel).astype(np.float32)
                            for kname, obs in (("fused", oF), ("packed", oP)):
                                # dump columns: 24 = elem(0,0), 25 = elem(1,0), 26 = elem(0,1), 27 = elem(1,1)
                                o_r0 = obs[sel][:, 0] if r == 0 else obs[sel][:, 1]
                                o_r1 = obs[sel][:, 2] if r == 0 else obs[sel][:, 3]
                                # e0 uses column index c=0 of row r; e1 uses c=1
                                m0 = e0.view(np.int32) == o_r0.view(np.int32)
                                m1 = e1.view(np.int32) == o_r1.view(np.int32)
                                score[kname][r, v0, v1, w0, w1] += int((m0 & m1).sum())
    print(f"\n  matched {n_samples} Gaussian x camera samples; fraction of samples whose elements (row r: both columns) equal the emulation:")
    for kname in ("fused", "packed"):
        for r in (0, 1):
            sc = score[kname][r].reshape(-1)
            order = np.argsort(-sc)[:3]
            print(f"  {kname:6} kernel, row {r} of R S R^T:")
            for rank, idx in enumerate(order):
                v0, v1, w0, w1 = np.unravel_index(idx, (3, 3, 3, 3))
                print(f"     #{rank + 1}: {sc[idx] / n_samples:7.4f}   (R S)[{r}][0]: {VARIANT[v0]}; (R S)[{r}][1]: {VARIANT[v1]}; (.. R^T)[{r}][0]: {VARIANT[w0]}; (.. R^T)[{r}][1]: {VARIANT[w1]}")


def three_term_experiment(a):
    """Pairing of the products in covarW2C for a general rotation (sums of three products), read position pair by pair.

    A sum x0*y0 + x1*y1 + x2*y2 in `R * covarW * transpose(R)` is evaluated as a chain of multiply-adds in some order.
    Zeroing the covariance outside the index block {p, q} makes the product at the third position exactly zero, so
    each sum reduces to two non-zero terms (positions p and q) and the two-term test says which of the two products
    is rounded first. Running the three blocks {0,1}, {0,2}, {1,2} gives the relative order of all three positions.
    Gaussians sit at world (0, 0, 2) with camera depth exactly 4 after the translation (camera-space x = y = 0 exactly,
    1/z = 0.25 exactly), so persp_proj afterwards is exact multiplications by 75.
    """
    dev = "cuda"
    mod = build(4)
    N, ncam, W, H, fx = 10000, 24, 256, 256, 300.0
    g = torch.Generator().manual_seed(4321)
    means = torch.zeros(N, 3)
    means[:, 2] = 2.0
    quats = torch.tensor([[1.0, 0, 0, 0]]).repeat(N, 1)
    scales = torch.ones(N, 3)
    opac = torch.full((N,), 0.9)
    K = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    to = lambda t: t.to(dev).contiguous()
    base = [to(means), to(quats), to(scales), to(opac)]

    cams = []
    while len(cams) < ncam:
        Rm = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))[0].float()
        vm = torch.eye(4)
        vm[:3, :3] = Rm
        vm[0, 3] = -2.0 * Rm[0, 2]
        vm[1, 3] = -2.0 * Rm[1, 2]
        vm[2, 3] = float(np.float32(4.0 - 2.0 * float(Rm[2, 2])))
        cams.append(vm)

    names = {0: "unfused", 1: "first product rounded, second fused", 2: "second product rounded, first fused"}
    obs_col = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
    for blk in ((0, 1), (0, 2), (1, 2)):
        pq = blk
        A = torch.randn(N, 2, 2, generator=g) * 0.1
        B = (A @ A.transpose(1, 2))
        B = ((B + B.transpose(1, 2)) / 2).float()
        Sfull = torch.zeros(N, 3, 3)
        for i in range(2):
            for j in range(2):
                Sfull[:, pq[i], pq[j]] = B[:, i, j]
        covars6 = torch.stack([Sfull[:, 0, 0], Sfull[:, 0, 1], Sfull[:, 0, 2], Sfull[:, 1, 1], Sfull[:, 1, 2], Sfull[:, 2, 2]], -1).contiguous()
        S_np = Sfull.numpy().astype(np.float32)
        score = {"unpacked (fused)": np.zeros((3, 3, 3, 3, 2), dtype=np.int64), "packed": np.zeros((3, 3, 3, 3, 2), dtype=np.int64)}
        n_samples, kept = 0, 0
        for vm in cams:
            args = base + [to(vm), to(K), W, H, 0.3, 0.01, 100.0, to(covars6)]
            dF = mod.dump_fused(*args)[0].cpu().numpy()
            dP = mod.dump_packed(*args)[0].cpu().numpy()
            if not all(((d[:, 28] == 0) & (d[:, 29] == 0) & (d[:, 30] == 4.0)).all() for d in (dF, dP)):
                continue
            kept += 1
            R = vm[:3, :3].numpy().astype(np.float32)
            n_samples += N
            p_, q_ = pq
            for r in (0, 1):
                # (R S)[r][j] = R[r][p] S[p][j] + R[r][q] S[q][j], j in {p, q}
                Mj = {j: two_term(R[r, p_], S_np[:, p_, j], R[r, q_], S_np[:, q_, j]) for j in (p_, q_)}
                for vp in range(3):
                    for vq in range(3):
                        # (R S R^T)[r][c] = M[r][p] R[c][p] + M[r][q] R[c][q]
                        sc = [two_term(Mj[p_][vp], R[c, p_], Mj[q_][vq], R[c, q_]) for c in (0, 1)]
                        for w0 in range(3):
                            for w1 in range(3):
                                e = [((np.float32(75) * sc[c][w]).astype(np.float32) * np.float32(75)).astype(np.float32).view(np.int32) for c, w in ((0, w0), (1, w1))]
                                for kname, d in (("unpacked (fused)", dF), ("packed", dP)):
                                    m = (e[0] == d[:, 24 + (0 if r == 0 else 1)].view(np.int32)) & (e[1] == d[:, 26 + (0 if r == 0 else 1)].view(np.int32))
                                    score[kname][vp, vq, w0, w1, r] += int(m.sum())
        print(f"\n  block {{{pq[0]},{pq[1]}}}: {kept} cameras, {n_samples} samples (row-wise, both columns of the 2x2 block must match)", flush=True)
        for kname in ("unpacked (fused)", "packed"):
            for r in (0, 1):
                sc = score[kname][..., r].reshape(-1)
                order = np.argsort(-sc)[:2]
                lines = []
                for idx in order:
                    vp, vq, w0, w1 = np.unravel_index(idx, (3, 3, 3, 3))
                    lines.append(f"{sc[idx] / n_samples:.4f} [(R S)[{r}][{pq[0]}]: {names[vp]}; (R S)[{r}][{pq[1]}]: {names[vq]}; (R S R^T)[{r}][0]: {names[w0]}; (R S R^T)[{r}][1]: {names[w1]}]")
                print(f"    {kname}, row {r}: best {lines[0]}", flush=True)
                print(f"        runner-up {lines[1]}", flush=True)


PERMS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def three_term_variants(a, b, c, d, e, f):
    """Eighteen ways to evaluate a*b + c*d + e*f in float32: 6 fused chains (product i rounded, then j and k fused in),
    3 with two products rounded and added and the third fused into that sum, 3 unfused sums in different orders, and
    6 with one fused pair (i rounded, j fused) followed by a plain add of the third rounded product."""
    P = [(a, b), (c, d), (e, f)]
    r = [(x * y).astype(np.float32) for x, y in P]
    out, names = [], []
    for i, j, k in PERMS:
        out.append(fma32(P[k][0], P[k][1], fma32(P[j][0], P[j][1], r[i])))
        names.append(f"p{i + 1} rounded, p{j + 1} fused, p{k + 1} fused")
    for k in range(3):
        i, j = [t for t in range(3) if t != k]
        out.append(fma32(P[k][0], P[k][1], (r[i] + r[j]).astype(np.float32)))
        names.append(f"p{i + 1}, p{j + 1} rounded and added, p{k + 1} fused into the sum")
    for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        out.append(((r[i] + r[j]).astype(np.float32) + r[k]).astype(np.float32))
        names.append(f"unfused: (p{i + 1} + p{j + 1}) + p{k + 1}")
    for i, j, k in PERMS:
        out.append((fma32(P[j][0], P[j][1], r[i]) + r[k]).astype(np.float32))
        names.append(f"p{i + 1} rounded, p{j + 1} fused, then p{k + 1} rounded and added")
    return out, names


def three_term_full(a):
    """Check the pattern of covarW2C on sums with three non-zero terms: one pattern for all sums, 18 x 18 candidates."""
    dev = "cuda"
    mod = build(4)
    N, ncam, W, H, fx = 20000, 24, 256, 256, 300.0
    g = torch.Generator().manual_seed(4321)
    A = torch.randn(N, 3, 3, generator=g) * 0.03
    S = A @ A.transpose(1, 2)
    S = ((S + S.transpose(1, 2)) / 2).float()
    covars6 = torch.stack([S[:, 0, 0], S[:, 0, 1], S[:, 0, 2], S[:, 1, 1], S[:, 1, 2], S[:, 2, 2]], -1).contiguous()
    means = torch.zeros(N, 3)
    means[:, 2] = 2.0
    base = [to_dev(t, dev) for t in (means, torch.tensor([[1.0, 0, 0, 0]]).repeat(N, 1), torch.ones(N, 3), torch.full((N,), 0.9))]
    K = torch.tensor([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
    S_np = S.numpy().astype(np.float32)
    cams = []
    while len(cams) < ncam:
        Rm = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))[0].float()
        vm = torch.eye(4)
        vm[:3, :3] = Rm
        vm[0, 3] = -2.0 * Rm[0, 2]
        vm[1, 3] = -2.0 * Rm[1, 2]
        vm[2, 3] = float(np.float32(4.0 - 2.0 * float(Rm[2, 2])))
        args = base + [to_dev(vm, dev), to_dev(K, dev), W, H, 0.3, 0.01, 100.0, to_dev(covars6, dev)]
        dF = mod.dump_fused(*args)[0].cpu().numpy()
        dP = mod.dump_packed(*args)[0].cpu().numpy()
        if all(((d[:, 28] == 0) & (d[:, 29] == 0) & (d[:, 30] == 4.0)).all() for d in (dF, dP)):
            cams.append((vm[:3, :3].numpy().astype(np.float32), dF[:, 24:28], dP[:, 24:28]))
    one = np.ones(1, np.float32)
    names = three_term_variants(one, one, one, one, one, one)[1]
    obs_col = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}

    def elem(sig):
        return ((np.float32(75) * sig).astype(np.float32) * np.float32(75)).astype(np.float32).view(np.int32)

    def evaluate(R, Ss, obs, pm, pc, Mv=None):
        """Fraction of samples whose four elements equal the emulation with pattern pm for R S and pc for (R S) R^T."""
        ok = np.ones(len(R), bool)
        for r in (0, 1):
            Mrow = [(Mv[r][j][pm] if Mv else three_term_variants(R[:, r, 0], Ss[:, 0, j], R[:, r, 1], Ss[:, 1, j], R[:, r, 2], Ss[:, 2, j])[0][pm]) for j in range(3)]
            for c in (0, 1):
                sg = three_term_variants(Mrow[0], R[:, c, 0], Mrow[1], R[:, c, 1], Mrow[2], R[:, c, 2])[0][pc]
                ok &= elem(sg) == obs[:, obs_col[(r, c)]].view(np.int32)
        return ok

    # search on a subset
    sub = cams[:6]
    nsub = 3000
    Rg = np.concatenate([np.broadcast_to(R, (nsub, 3, 3)) for R, _, _ in sub])
    Sg = np.concatenate([S_np[:nsub]] * len(sub))
    print(f"  {len(cams)} cameras kept, searching {len(Rg)} samples over 18 x 18 patterns", flush=True)
    for kname, col in (("unpacked (fused)", 1), ("packed", 2)):
        obs = np.concatenate([c[col] [:nsub] for c in sub])
        Mv = [[three_term_variants(Rg[:, r, 0], Sg[:, 0, j], Rg[:, r, 1], Sg[:, 1, j], Rg[:, r, 2], Sg[:, 2, j])[0] for j in range(3)] for r in (0, 1)]
        res = []
        for pm in range(18):
            for pc in range(18):
                res.append((evaluate(Rg, Sg, obs, pm, pc, Mv).mean(), pm, pc))
        res.sort(reverse=True)
        print(f"  {kname}:", flush=True)
        for f, pm, pc in res[:3]:
            print(f"     {f:.4f}  (R S): {names[pm]};  (R S) R^T: {names[pc]}", flush=True)
        f, pm, pc = res[0]
        tot = ok_n = 0
        for R, dFo, dPo in cams:
            Rb = np.broadcast_to(R, (N, 3, 3))
            e = evaluate(Rb, S_np, dFo if col == 1 else dPo, pm, pc)
            tot += len(e)
            ok_n += int(e.sum())
        print(f"     best pattern on all {tot} samples (all four elements): {100 * ok_n / tot:.3f}%", flush=True)

    # The block tests fix the order of the three products only for entries that see all three position pairs
    # ((R S)[r][0] sees (0,1) and (0,2); (R S)[r][1] sees (0,1) and (1,2); (R S)[r][2] sees (0,2) and (1,2); R S R^T sees all),
    # so each entry may use a different chain. Search the six fused chains independently for each of the three entries of
    # row r of R S and for each of the two entries of R S R^T.
    from itertools import product as iproduct

    print("  per-entry fused chain (index into the six orders: " + "; ".join(f"{i}: {names[i]}" for i in range(6)) + "):", flush=True)
    use = cams[:6]
    nuse = 3000
    Ru = np.concatenate([np.broadcast_to(R, (nuse, 3, 3)) for R, _, _ in use])
    Su = np.concatenate([S_np[:nuse]] * len(use))
    found = {}
    for kname, col in (("unpacked (fused)", 1), ("packed", 2)):
        obs = np.concatenate([c[col][:nuse] for c in use])
        for r in (0, 1):
            Mv = [three_term_variants(Ru[:, r, 0], Su[:, 0, j], Ru[:, r, 1], Su[:, 1, j], Ru[:, r, 2], Su[:, 2, j])[0][:6] for j in range(3)]
            results = []
            for mo in iproduct(range(6), repeat=3):
                Mrow = [Mv[j][mo[j]] for j in range(3)]
                sc = [three_term_variants(Mrow[0], Ru[:, c, 0], Mrow[1], Ru[:, c, 1], Mrow[2], Ru[:, c, 2])[0][:6] for c in (0, 1)]
                for so in iproduct(range(6), repeat=2):
                    ok = np.ones(len(Ru), bool)
                    for c in (0, 1):
                        ok &= elem(sc[c][so[c]]) == obs[:, obs_col[(r, c)]].view(np.int32)
                    results.append((ok.mean(), mo, so))
            results.sort(reverse=True)
            f, mo, so = results[0]
            found[(kname, r)] = (mo, so)
            print(f"    {kname}, row {r}: best {f:.4f} with chains (R S)[{r}][0..2] = {list(mo)}, (R S R^T)[{r}][0..1] = {list(so)}; runner-up {results[1][0]:.4f}", flush=True)
    print("  checked on every kept camera and Gaussian (all four elements):", flush=True)
    for kname, col in (("unpacked (fused)", 1), ("packed", 2)):
        tot = ok_n = 0
        for R, dFo, dPo in cams:
            obs = dFo if col == 1 else dPo
            Rb = np.broadcast_to(R, (N, 3, 3))
            ok = np.ones(N, bool)
            for r in (0, 1):
                mo, so = found[(kname, r)]
                Mrow = [three_term_variants(Rb[:, r, 0], S_np[:, 0, j], Rb[:, r, 1], S_np[:, 1, j], Rb[:, r, 2], S_np[:, 2, j])[0][mo[j]] for j in range(3)]
                for c in (0, 1):
                    sg = three_term_variants(Mrow[0], Rb[:, c, 0], Mrow[1], Rb[:, c, 1], Mrow[2], Rb[:, c, 2])[0][so[c]]
                    ok &= elem(sg) == obs[:, obs_col[(r, c)]].view(np.int32)
            tot += N
            ok_n += int(ok.sum())
        print(f"    {kname}: {100 * ok_n / tot:.3f}% of {tot} samples reproduced", flush=True)


def to_dev(t, dev):
    return t.to(dev).contiguous()


def covars_experiment(a):
    """Feed both real kernels the same precomputed covariance, so quat_scale_to_covar_preci is skipped."""
    tot = {"n": 0, "diff": 0, "diff_quats": 0}
    for i in range(a.n_scenes):
        _, scene, cams = load(a, i)
        cam = cams[0]
        covars, _ = gsplat.quat_scale_to_covar_preci(scene.quats, scene.scales, compute_covar=True, compute_preci=False, triu=False)
        res = {}
        for packed in (False, True):
            for use in ("covars", "quats"):
                kw = dict(covars=covars) if use == "covars" else {}
                _, _, meta = gsplat.rasterization(
                    means=scene.means, quats=scene.quats, scales=scene.scales, opacities=scene.opacities, colors=scene.colors,
                    viewmats=cam.viewmat[None], Ks=cam.K[None], width=cam.width, height=cam.height, near_plane=0.01,
                    far_plane=100.0, sh_degree=None, packed=packed, **kw,
                )
                res[(packed, use)] = dense(meta, scene.n, packed)
        (du, vu), (dp, vp) = res[(False, "covars")], res[(True, "covars")]
        ok = vu & vp
        tot["n"] += int(ok.sum())
        tot["diff"] += int((bits(du["conics"][ok]) != bits(dp["conics"][ok])).any(-1).sum())
        tot["diff_quats"] += int((bits(res[(False, "quats")][0]["conics"][ok]) != bits(res[(True, "quats")][0]["conics"][ok])).any(-1).sum())
    n = tot["n"]
    print(f"\n=== precomputed covariance fed to both kernels ({a.n_scenes} scenes, {n} Gaussians)")
    print(f"  packed vs unpacked conics differ with quats+scales input:  {tot['diff_quats']} ({100*tot['diff_quats']/n:.1f}%)")
    print(f"  packed vs unpacked conics differ with precomputed covars:  {tot['diff']} ({100*tot['diff']/n:.1f}%)")


def identity_rotation_experiment(a):
    """Camera rotation = identity (translation only): covarW2C = R S R^T is then exact (1s and 0s), so any
    remaining packed-vs-unpacked difference comes from persp_proj and later."""
    tot = {"n": 0, "diff": 0, "diff_rot": 0, "n_rot": 0}
    for i in range(a.n_scenes):
        _, scene, cams = load(a, i)
        cam = cams[0]
        covars, _ = gsplat.quat_scale_to_covar_preci(scene.quats, scene.scales, compute_covar=True, compute_preci=False, triu=False)
        viewmat = torch.eye(4, device=scene.means.device)
        viewmat[2, 3] = 3.0 * scene.means.norm(dim=-1).max() + 1.0
        res = {}
        for packed in (False, True):
            for tag, vm in (("id", viewmat), ("rot", cam.viewmat)):
                _, _, meta = gsplat.rasterization(
                    means=scene.means, quats=scene.quats, scales=scene.scales, opacities=scene.opacities, colors=scene.colors,
                    viewmats=vm[None], Ks=cam.K[None], width=cam.width, height=cam.height, near_plane=0.01, far_plane=100.0,
                    sh_degree=None, packed=packed, covars=covars,
                )
                res[(packed, tag)] = dense(meta, scene.n, packed)
        for tag, key_n, key_d in (("id", "n", "diff"), ("rot", "n_rot", "diff_rot")):
            (du, vu), (dp, vp) = res[(False, tag)], res[(True, tag)]
            ok = vu & vp
            tot[key_n] += int(ok.sum())
            tot[key_d] += int((bits(du["conics"][ok]) != bits(dp["conics"][ok])).any(-1).sum())
    print(f"\n=== identity camera rotation, precomputed covariance fed to both kernels ({a.n_scenes} scenes)")
    print(f"  packed vs unpacked conics differ, identity rotation:  {tot['diff']} of {tot['n']} ({100*tot['diff']/max(tot['n'],1):.1f}%)")
    print(f"  packed vs unpacked conics differ, the scene's own camera rotation: {tot['diff_rot']} of {tot['n_rot']} ({100*tot['diff_rot']/max(tot['n_rot'],1):.1f}%)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-scenes", type=int, default=100)
    p.add_argument("--stages", type=int, nargs="*", default=[0, 1, 2, 3])
    p.add_argument("--covars", action="store_true", help="run the precomputed-covariance experiment first")
    p.add_argument("--pairing", action="store_true", help="run the rounding-pattern experiment for covarW2C first")
    p.add_argument("--three-term", action="store_true", help="run the position-pair rounding-pattern experiment for covarW2C")
    p.add_argument("--three-term-full", action="store_true", help="check the rounding pattern of covarW2C on sums of three non-zero terms")
    p.add_argument("--identity-rot", action="store_true", help="run the identity-camera-rotation experiment first")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    a.large, a.mode, a.scene_idx = True, None, None
    if a.three_term_full or a.three_term or a.pairing:
        print(f"GPU: {torch.cuda.get_device_name(0)}, compute capability {torch.cuda.get_device_capability(0)}, torch {torch.__version__}, CUDA {torch.version.cuda}", flush=True)
    if a.three_term_full:
        print("\n=== rounding pattern of covarW2C on sums of three non-zero terms, one pattern for all sums", flush=True)
        three_term_full(a)
    if a.three_term:
        print("\n=== rounding pattern of covarW2C with three non-zero terms per sum (Gaussians on the optical axis, general rotation)")
        three_term_experiment(a)
    if a.pairing:
        print("\n=== rounding pattern of covarW2C in the two kernels (Gaussians on the optical axis, rotation about z)")
        pairing_experiment(a)
    if a.identity_rot:
        identity_rotation_experiment(a)
    if a.covars:
        covars_experiment(a)
    for st in a.stages:
        run_stage(a, st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
