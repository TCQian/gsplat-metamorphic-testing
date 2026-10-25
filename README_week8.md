# Week 8: MR4/MR5 and render-visibility checks

World-scale and permutation relations, checks for renders that prove nothing, and a diagnosis of the one MR4 discrepancy found.

## Changes

### MR4: uniform world-scale invariance (`gsplat_mt/mr4_world_scale.py`)

`run_mr4(scene, camera, k, mode)` renders the scene, then renders it again with means, scales, camera translation and near/far planes multiplied by `k` (intrinsics, rotations, colors, opacities unchanged).

- `correct`: full joint transform.
- `no_translation`: fault injection; camera translation left unscaled.

The runner sweeps `--world-scales` (default `0.01 0.25 4 100`) and records the worst case (`mr4_rgb_d_max`, `mr4_worst_k`). The fault injection runs at the largest factor.

### MR5: Gaussian permutation invariance (`gsplat_mt/mr5_permutation.py`)

`run_mr5(scene, camera, seed, mode)` applies one seeded random permutation to every per-Gaussian attribute and compares renders.

- `correct`: all attributes permuted together.
- `misaligned`: fault injection; colors left unpermuted.
- `mr5_depth_tie`: set when Gaussians in front of the camera have depths closer than 1e-6. A difference in a tied scene is reported as `tie_ambiguous`, not `violation`.

### Oracle (`gsplat_mt/metrics.py`)

- `TOLERANCES` adds `mr4: 1e-4` (same as MR1, since every float is rescaled) and `mr5: 0` (order should not matter away from depth ties).
- `compare_images(a, b, tol)` also counts `n_px_over`: pixels whose max channel difference exceeds `tol` (CSV: `*_n_px_over`).
- `verdict()` can return `isolated_discontinuity` for MR1 and MR4 only: every image over tolerance has at most 0.05% of its pixels over tolerance (`ISOLATED_PIXEL_FRAC`; 2 of 4096 at 64x64). MR2, MR3, MR5 and the fault-injection rows are never relabeled, since those should be bit-exact or must keep detecting bugs.

### Render visibility (`gsplat_mt/run_preliminary.py`)

`render_stats` renders each scene once across its cameras and records:

- `coverage_cam0`, `coverage_mean`: fraction of pixels with alpha > 0.
- `opaque_frac_cam0`, `opaque_frac_mean`: fraction of pixels with alpha > 0.99.
- `n_empty_cams`: cameras that drew nothing.
- `n_visible_cam0`, `n_visible_any`: Gaussians surviving gsplat's culling (`meta["radii"] > 0`).

A `pass` on a render that draws nothing becomes `vacuous_pass`: camera 0 empty for MR1, MR3, MR4, MR5 (and their fault-injection rows); every camera empty for MR2.

### Generator

`scale_huge` scales narrowed from [0.5, 2.0] to [0.15, 0.5] (see Findings).

### Diagnosis tool (`gsplat_mt/diagnose_mr4.py`)

Replays one scene and compares original vs k-scaled renders pixel by pixel, over power-of-two k (exact in binary) and inexact k. For pixels over tolerance it reports where they are and the alpha of the single Gaussian that would explain the difference (`alpha = 1 - T_small / T_large`, from the final transmittance at that pixel). Run as `python gsplat_mt/diagnose_mr4.py --scene-idx N --mode MODE`.

## What was tested

8 modes x 20 random scenes (160 scenes; 20-200 Gaussians, 64x64, 4 cameras) through MR1-MR5 with all three fault injections (MR3 `xyzw_bug`, MR4 `no_translation`, MR5 `misaligned`), plus the issue #764 and #630 reproductions.

## Findings

| Relation                  | Max D_max | Verdict                                           |
| ------------------------- | --------- | ------------------------------------------------- |
| MR1 packed vs unpacked    | 3.0e-07   | pass                                              |
| MR2 batched vs individual | 0         | pass                                              |
| MR3 covariance (correct)  | 0         | pass                                              |
| MR4 world scale           | 6.0e-04   | pass, except one `isolated_discontinuity` (below) |
| MR5 permutation           | 0         | pass                                              |

- No scene crashed. Fault injections were detected in every scene with a non-empty render (D_max up to 0.59 for MR3, 0.96 for MR4, 0.92 for MR5). MR5's injection cannot fire for `single` (N = 1), where the permutation is the identity.
- MR4's worst case was always k = 100 or k = 0.01. MR4 noise is about 1e-6 for typical scenes.
- MR5: 3 scenes had near-equal depths; none produced a difference.
- One `far_clipped` scene rendered nothing; its results are `vacuous_pass`. Its `n_visible_cam0` was 1, so the culling-based count does not guarantee drawn pixels; coverage is what verdicts use.

### The MR4 discrepancy is a rasterizer threshold flip, not a scale bug

One dense `scale_huge` scene (188 Gaussians) exceeded the MR4 tolerance at k = 0.01 (D_max 5.96e-04 against 1e-4, but D_mae 1.4e-07). Per-pixel diagnosis:

- Power-of-two k (1/128, 1/4, 4, 128) gave D_max exactly 0 over all pixels, and repeated renders are identical. gsplat's scale handling is exact.
- At k = 0.01 about 3800 of 4096 pixels differed by ~1e-7 (rounding from the inexact factor), and exactly one pixel differed by 5.96e-04.
- At that pixel, final transmittance differed by 0.4%, equal to `(1 - 1/255)^-1`: one Gaussian with alpha near 1/255 was skipped in one render and blended in the other. gsplat skips Gaussians with alpha < 1/255 (`ALPHA_THRESHOLD` in `cuda/include/Common.h`). Per-Gaussian alpha at that pixel was not logged, so this is strongly supported but not directly observed.
- Under the isolated-pixel rule this scene is reported as `isolated_discontinuity` (1 of 4096 pixels over tolerance), with D_max still recorded. It is the only scene in the run with a nonzero MR4 pixel count; MR1 pixel counts are 0 everywhere.

### Image content: saturation and sparsity

- With the old `scale_huge` range [0.5, 2.0], every pixel had alpha > 0 and the median opaque fraction was 1.0 (18 of 20 scenes over 95% opaque), so Gaussians behind the front layer could not affect pixels. With [0.15, 0.5] the median opaque fraction is 0.16 (range 0.08-0.44) and median coverage 0.79.
- In the other modes the opaque fraction is at most 0.05 and median coverage is 0.01-0.16, so most pixels are background. Passes there rest on sparse images; the injected bugs are still detected because they move enough pixels.
