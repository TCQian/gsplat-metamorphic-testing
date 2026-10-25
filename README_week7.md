# Week 7: scene generator and oracle refinements

Stress modes for the scene generator, a tolerance-based pass/violation oracle, and per-scene timing.

## Changes

### Scene generator (`gsplat_mt/scene.py`)

`sample_scene(mode, n, ...)` wraps `sample_gaussians` with a stress mode. `SCENE_MODES`:

| Mode              | Effect                                                                             |
| ----------------- | ---------------------------------------------------------------------------------- |
| `default`         | unchanged generator                                                                |
| `opacity_extreme` | opacities drawn from {0, 1e-4, 0.999, 1}                                           |
| `scale_tiny`      | scales in [1e-4, 1e-3]                                                             |
| `scale_huge`      | scales in [0.5, 2.0] (narrowed later in week 8)                                    |
| `anisotropic`     | one random axis per Gaussian shrunk 1000x                                          |
| `wide_spread`     | means in a radius-8 ball (Gaussians behind camera, off-frustum, inside near plane) |
| `far_clipped`     | radius-150 ball, scales in [1, 3] (most beyond `far_plane`)                        |
| `single`          | N = 1                                                                              |

Mode overrides take precedence over the caller's sampling arguments. There is no N = 0 mode: a CUDA illegal-memory-access error is sticky and would corrupt every later scene in the process.

### Oracle (`gsplat_mt/metrics.py`)

- `compare_images` is NaN/Inf aware: a NaN/Inf mismatch between the two images gives `d_max = inf` instead of being dropped by `max()`.
- `TOLERANCES` (max abs pixel difference), set from the noise floor measured on benign scenes: MR1 `1e-4`, MR2 `0`, MR3 `0`.
- `verdict(diffs, error, tol)` returns `crash | violation | invalid_output | pass`.
- For fault-injected variants (MR3 `xyzw_bug`), `violation` means the injected bug was detected.

### Runner (`gsplat_mt/run_preliminary.py`)

- `--n-scenes` is per mode; `--modes` selects modes.
- CSV has a `mode` column, one verdict column per relation, and `time_s` (wall-clock seconds per scene, CUDA-synchronized; the first scene includes the one-time CUDA extension compile).
- `*_summary.json` holds `stats` (D_max/D_MAE min/mean/max) and `verdicts` (counts per mode).
- Default output is `results/refined_results.csv`.

## What was tested

8 modes x 20 random scenes (160 scenes; 20-200 Gaussians, 64x64, 4 cameras), each run through MR1 (packed vs unpacked), MR2 (batched vs individual cameras), MR3 (covariance vs scale+quaternion, plus the `xyzw_bug` fault injection), and the standalone reproductions of upstream gsplat issues #764 and #630.

## Findings

- No scene crashed or produced NaN/Inf.
- MR1: max D_max 2.4e-07, far below the 1e-4 tolerance (float noise).
- MR2 and MR3 (correct): D_max exactly 0 in every scene and every mode.
- MR3 `xyzw_bug`: detected in 159 of 160 scenes (D_max up to 0.59). The one miss was a `far_clipped` scene whose render contained nothing, so no wrong rotation could change the image (confirmed in week 8 with coverage statistics).
- Issues #764 (`packed=True` with `backgrounds` asserts) and #630 (quaternion convention mismatch, a caller-side bug) both reproduce.

## Limitations

- The CSV did not yet record whether a render drew anything, so a `pass` on `wide_spread` or `far_clipped` could not be told apart from an empty render. Week 8 adds this.
