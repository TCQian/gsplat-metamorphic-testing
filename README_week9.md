# Week 9: large-scale randomized testing

Adds a runner that draws a fresh configuration for every scene. Not yet run on the cluster.

## Changes

### Runner (`gsplat_mt/run_large_scale.py`)

Each scene samples a configuration, then runs the same per-scene pipeline as `run_preliminary.py` (`run_one_scene`: MR1-MR5, fault injections, visibility checks, verdicts). Sampling is deterministic from `--seed` and independent of the scene's own seed.

| Parameter | Distribution |
| --- | --- |
| Gaussians N | log-uniform, `--n-min` 10 to `--n-max` 5000 |
| Scene radius | log-uniform, 0.3 to 2.0 |
| Scale range | center log-uniform 0.01 to 0.3; range is 0.5x to 1.5x the center |
| Opacity range | low uniform 0.05 to 0.9; high uniform from low to 1.0 |
| Camera distance | min = radius x uniform(1.2, 8); max = min x uniform(1.0, 1.6) |
| Field of view | uniform 30 to 90 degrees |
| Cameras | 1 to `--max-cameras` (default 8) |
| Resolution | one of 64x64, 120x90, 128x128, 256x192, 256x256 |
| Mode | uniform over `--modes` (default all 8 scene modes) |

A mode's own overrides win over the sampled values they touch (for example `wide_spread` fixes the scene radius), so `cfg_*` columns do not apply to those fields in that mode.

### Flags

- `--n-scenes` (default 2000), `--seed` (default 1000), `--modes`, `--n-min/--n-max`, `--max-cameras`, `--world-scales`, `--out`.
- `--only-scene i`: replay one scene index with its sampled configuration.

### Outputs (next to `--out`)

- `*.csv`: all scenes; sampled configuration in `cfg_*` columns (`cfg_pixels` = width x height).
- `*.jsonl`: one row per scene, flushed as it goes, so a timeout keeps partial results.
- `*_findings.csv`: scenes with a crash, or any MR1-MR5 verdict other than `pass` / `vacuous_pass` (violation, `isolated_discontinuity`, `tie_ambiguous`, `invalid_output`).
- `*_summary.json`: `n_scenes`, `aborted`, `n_findings`, `stats` (D_max min/mean/max and `time_s`), `verdicts` per mode, `fault_injection` outcome counts, and `spearman` (rank correlation of `n_gaussians`, `n_cameras`, `mean_scale`, `mean_opacity`, `mean_depth`, `overlap_frac`, `coverage_cam0`, `opaque_frac_cam0`, `cfg_pixels` with `mr1_rgb_d_max` and `mr4_rgb_d_max`; `null` where a column is constant or has fewer than 10 usable scenes).

### Failure handling

- A scene that raises is recorded with `fatal_error` and the run continues.
- Out-of-memory errors empty the CUDA cache and continue.
- Any other CUDA error stops the run (exit code 1, `aborted` set in the summary), since a sticky CUDA error would invalidate every later scene.

### SLURM

`soc-cluster-large.sh`: one `h100-96` GPU, 3 hours, `--n-scenes 2000 --out results/large_scale.csv`.

## Checked so far

- Files compile.
- `sample_config` is deterministic over 2000 scenes and stays in the ranges above; modes and resolutions are drawn roughly evenly; no scene has `opacity_max < opacity_min`, `dist_max < dist_min`, or a camera inside the scene radius.
- Not checked: any GPU execution, per-scene time at N = 5000 with 256x256 and 8 cameras (so the 3-hour limit is a guess), and the Spearman code.
