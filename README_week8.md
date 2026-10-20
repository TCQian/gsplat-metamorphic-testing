# Week 8: MR4/MR5 and render-visibility checks

Changes since the week 7 commit, and the results of the resulting run (SLURM job 979783).

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

`verdict()` can return `isolated_discontinuity` (MR1 and MR4 only): every image over tolerance has at most `ISOLATED_PIXEL_FRAC = 5e-4` of its pixels over tolerance (2 of 4096 at 64x64). `compare_images(a, b, tol)` now also counts `n_px_over` (pixels whose max channel difference exceeds `tol`), written to the CSV as `*_n_px_over`. MR2, MR3, MR5 and the fault-injection rows are never relabeled.

`TOLERANCES` now has `mr4: 1e-4` (same as MR1, since every float is rescaled) and `mr5: 0.0` (order should not matter away from depth ties). Both are initial values.

### Render visibility (`gsplat_mt/run_preliminary.py`)

`render_stats` renders each scene once across its cameras and records:

- `coverage_cam0`, `coverage_mean`: fraction of pixels with alpha > 0.
- `opaque_frac_cam0`, `opaque_frac_mean`: fraction of pixels with alpha > 0.99.
- `n_empty_cams`: cameras that drew nothing.
- `n_visible_cam0`, `n_visible_any`: Gaussians with `meta["radii"] > 0` (gsplat's culling).

A `pass` on a render that draws nothing becomes `vacuous_pass`: camera 0 empty for MR1, MR3, MR4, MR5 (and their fault-injection rows); every camera empty for MR2.

### Generator

`scale_huge` scales shrunk from [0.5, 2.0] to [0.15, 0.5]; at the old range 18/20 scenes were over 95% opaque (see Notes).

### Runner

- New CSV columns for MR4/MR5 and the visibility stats; new verdict keys `mr4_verdict`, `mr4_bug_verdict`, `mr5_verdict`, `mr5_bug_verdict`.
- `gsplat_mt/diagnose_mr4.py`: per-pixel MR4 diagnosis for one scene (`--scene-idx`, `--mode`), over exact (power-of-two) and inexact k.
- `soc-cluster.sh` writes to `results/week8_results.csv`.

## Results (job 979783: 8 modes x 20 scenes = 160; MR4 row updated from job 981409)

Job 979783 used the old `scale_huge` range; job 981409 reran with [0.15, 0.5]. Results were identical except for `scale_huge` and the MR1/MR4 noise maxima.

| Relation | Max D_max (979783 / 981409) | Verdict |
| --- | --- | --- |
| MR1 packed vs unpacked | 2.38e-07 / 2.98e-07 | pass |
| MR2 batched vs individual | 0 / 0 | pass |
| MR3 covariance (correct) | 0 / 0 | pass |
| MR4 world scale | 4.68e-06 / 5.96e-04 | pass / 1 violation (scene 68) |
| MR5 permutation | 0 / 0 | pass |

- Scenes with an MR error or crash: 0/160.
- Fault injections: MR3 `xyzw_bug` (D_max up to 0.593), MR4 `no_translation` (up to 0.961) and MR5 `misaligned` (up to 0.922) were detected in every scene with a non-empty render.
- MR4's worst case was always k=100 or k=0.01.
- Job 981409: `scale_huge` scene 68 (188 Gaussians) violates MR4 at k=0.01: D_max 5.96e-04 (tolerance 1e-4) but D_mae 1.35e-07, i.e. a difference confined to very few pixel values. The next largest MR4 D_max in the run is 4.68e-06. `gsplat_mt/diagnose_mr4.py` (run on job 985185) shows the cause is a rasterizer threshold flip, not a scale-handling bug:
  - Power-of-two k (1/128, 1/4, 4, 128) give D_max exactly 0 over all pixels; repeated renders are identical.
  - At k=0.01 about 3800 of 4096 pixels differ by ~1e-7 (float noise) and exactly one pixel, (y=19, x=17), differs by 5.96e-04. Its remaining transmittance (1-alpha) is 4.206e-3 vs 4.189e-3, a 0.4% ratio equal to (1 - 1/255)^-1, i.e. one Gaussian with alpha near 1/255 was skipped in one render and blended in the other.
  - gsplat 1.5.3 skips Gaussians with alpha < 1/255 (`ALPHA_THRESHOLD` in `cuda/include/Common.h`). Per-Gaussian alpha at that pixel was not logged, so this is strongly supported but not directly observed.
- MR5 depth ties: 3 scenes, none produced a `tie_ambiguous`.
- Issues #764 and #630 reproduced again.
- Per-scene time: mean 15 ms, max 0.23 s (CUDA extension already compiled).

## Notes

- `far_clipped` scene 138 (68 Gaussians) rendered nothing. Its MR results are `vacuous_pass`. It reports `n_visible_cam0 = 1`, so the radii-based count does not guarantee drawn pixels; coverage is what the verdicts use.
- `far_clipped` has a median of 6 visible Gaussians out of 132; `wide_spread` 17.5 out of 85.
- With the old `scale_huge` range [0.5, 2.0], `coverage_cam0 = 1.0` in every scene and `opaque_frac_cam0` had median 1.000 (18/20 scenes > 0.95), so hidden Gaussians could not affect pixels. With [0.15, 0.5] (job 981409): `opaque_frac_cam0` median 0.161 (0.080-0.442, 0/20 > 0.95), `coverage_cam0` median 0.786 (min 0.458).
- In the other modes `opaque_frac_cam0` is at most 0.051 and median `coverage_cam0` is 0.012-0.156, so most pixels are background.
- MR5's fault injection cannot fire for `single` (N = 1), since the permutation is the identity.
