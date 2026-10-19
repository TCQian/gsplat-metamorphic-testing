# Week 7: scene generator and oracle refinements

Changes in commit `58509c8` ("refine preliminary results"), and the results of the resulting run (SLURM job 974275).

## Changes

### Scene generator (`gsplat_mt/scene.py`)

`sample_scene(mode, n, ...)` wraps `sample_gaussians` with a stress mode. `SCENE_MODES`:

| Mode | Effect |
| --- | --- |
| `default` | unchanged generator |
| `opacity_extreme` | opacities drawn from {0, 1e-4, 0.999, 1} |
| `scale_tiny` | scales in [1e-4, 1e-3] |
| `scale_huge` | scales in [0.5, 2.0] |
| `anisotropic` | one random axis per Gaussian shrunk 1000x |
| `wide_spread` | means in a radius-8 ball (Gaussians behind camera, off-frustum, inside near plane) |
| `far_clipped` | radius-150 ball, scales in [1, 3] (most beyond `far_plane`) |
| `single` | N = 1 |

Mode overrides take precedence over the caller's sampling arguments.

### Oracle (`gsplat_mt/metrics.py`)

- `compare_images` is NaN/Inf aware: a NaN/Inf mismatch between the two images gives `d_max = inf` instead of being dropped by `max()`.
- `TOLERANCES` (max abs pixel difference): MR1 `1e-4`, MR2 `0`, MR3 `0`.
- `verdict(diffs, error, tol)` returns `crash | violation | invalid_output | pass`.
- For the MR3 `xyzw_bug` fault injection, `violation` means the injected bug was detected.

### Runner (`gsplat_mt/run_preliminary.py`)

- `--n-scenes` is now per mode; new `--modes` flag.
- CSV has a `mode` column and `mr1_verdict`, `mr2_verdict`, `mr3_correct_verdict`, `mr3_xyzwbug_verdict` columns.
- `*_summary.json` is now `{"stats": ..., "verdicts": {mode: {verdict_key: {verdict: count}}}}`.
- CSV has a `time_s` column (wall-clock seconds per scene, CUDA-synchronized), also printed per scene and included in the summary stats. The first scene includes the one-time CUDA extension JIT compile.
- Default output is `results/refined_results.csv`; `soc-cluster.sh` updated to match.

### Cleanup

- Removed the reference to a non-existent `scripts/selfcheck_geometry.py` in `scene.py`.
- Usage line and CUDA-missing error text no longer reference `python -m` / Colab.

## Results (job 974275: 8 modes x 20 scenes = 160)

- Scenes with an MR error or crash: 0/160.
- MR1 D_max: max 2.38e-07 (below the 1e-4 tolerance).
- MR2 D_max: exactly 0 in all scenes.
- MR3 (correct): exactly 0 in all scenes.
- MR1, MR2 and MR3 (correct) verdicts: `pass` in all 160 scenes, in every mode.
- MR3 `xyzw_bug`: detected in 159/160 scenes (D_max up to 0.593). The one miss is `far_clipped` scene 138 (D_max = 0.0).
- Issue #764 and #630 reproductions: both detected (`results/refined_results_repro.json`).

## Notes

- `far_clipped` scene 138 (68 Gaussians) missing the injected bug suggests the render contained no visible Gaussians, so its MR passes are vacuous. This is not confirmed; the CSV does not record how many Gaussians are visible or the image coverage.
- `wide_spread` and `far_clipped` likewise do not record visible-Gaussian counts, so their MR passes cannot be told apart from empty renders.
- No N = 0 mode: a CUDA illegal-memory-access error is sticky and would corrupt all subsequent scenes in the process.
