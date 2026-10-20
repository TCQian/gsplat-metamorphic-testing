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

`TOLERANCES` now has `mr4: 1e-4` (same as MR1, since every float is rescaled) and `mr5: 0.0` (order should not matter away from depth ties). Both are initial values.

### Render visibility (`gsplat_mt/run_preliminary.py`)

`render_stats` renders each scene once across its cameras and records:

- `coverage_cam0`, `coverage_mean`: fraction of pixels with alpha > 0.
- `opaque_frac_cam0`, `opaque_frac_mean`: fraction of pixels with alpha > 0.99.
- `n_empty_cams`: cameras that drew nothing.
- `n_visible_cam0`, `n_visible_any`: Gaussians with `meta["radii"] > 0` (gsplat's culling).

A `pass` on a render that draws nothing becomes `vacuous_pass`: camera 0 empty for MR1, MR3, MR4, MR5 (and their fault-injection rows); every camera empty for MR2.

### Runner

- New CSV columns for MR4/MR5 and the visibility stats; new verdict keys `mr4_verdict`, `mr4_bug_verdict`, `mr5_verdict`, `mr5_bug_verdict`.
- `soc-cluster.sh` writes to `results/week8_results.csv`.

## Results (job 979783: 8 modes x 20 scenes = 160)

| Relation | Max D_max | Verdict |
| --- | --- | --- |
| MR1 packed vs unpacked | 2.38e-07 | pass |
| MR2 batched vs individual | 0 | pass |
| MR3 covariance (correct) | 0 | pass |
| MR4 world scale | 4.68e-06 | pass |
| MR5 permutation | 0 | pass |

- Scenes with an MR error or crash: 0/160.
- Fault injections: MR3 `xyzw_bug` (D_max up to 0.593), MR4 `no_translation` (up to 0.961) and MR5 `misaligned` (up to 0.922) were detected in every scene with a non-empty render.
- MR4's worst case was always k=100 or k=0.01.
- MR5 depth ties: 3 scenes, none produced a `tie_ambiguous`.
- Issues #764 and #630 reproduced again.
- Per-scene time: mean 15 ms, max 0.23 s (CUDA extension already compiled).

## Notes

- `far_clipped` scene 138 (68 Gaussians) rendered nothing. Its MR results are `vacuous_pass`. It reports `n_visible_cam0 = 1`, so the radii-based count does not guarantee drawn pixels; coverage is what the verdicts use.
- `far_clipped` has a median of 6 visible Gaussians out of 132; `wide_spread` 17.5 out of 85.
- `scale_huge` has `coverage_cam0 = 1.0` in every scene (alpha > 0 on every pixel). How opaque those pixels are was not recorded in that run; `opaque_frac_*` was added afterwards and has not been run yet.
- MR5's fault injection cannot fire for `single` (N = 1), since the permutation is the identity.
