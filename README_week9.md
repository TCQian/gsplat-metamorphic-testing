# Week 9: large-scale randomized testing

A runner that draws a fresh scene configuration for every scene, and the results of 2000 randomized scenes through MR1-MR5.

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

Flags: `--n-scenes` (default 2000), `--seed` (default 1000), `--modes`, `--n-min/--n-max`, `--max-cameras`, `--world-scales`, `--out`, and `--only-scene i` to replay one scene index with its sampled configuration.

### Outputs (next to `--out`)

- `*.csv`: all scenes; sampled configuration in `cfg_*` columns (`cfg_pixels` = width x height); `time_s` is wall-clock seconds per scene (CUDA-synchronized).
- `*.jsonl`: one row per scene, flushed as it goes, so a timeout keeps partial results.
- `*_findings.csv`: scenes with a crash, or any MR1-MR5 verdict other than `pass` / `vacuous_pass` (`violation`, `isolated_discontinuity`, `tie_ambiguous`, `invalid_output`).
- `*_summary.json`: `n_scenes`, `aborted`, `n_findings`, `timing` (total wall-clock; first-scene time reported separately because it includes the CUDA compile; median/p95/p99/max and per-mode median for the rest), `stats` (D_max and `time_s` min/mean/max), `verdicts` per mode, `fault_injection` outcome counts, and `spearman` (rank correlation of `n_gaussians`, `n_cameras`, `mean_scale`, `mean_opacity`, `mean_depth`, `overlap_frac`, `coverage_cam0`, `opaque_frac_cam0`, `cfg_pixels` with `mr1_rgb_d_max` and `mr4_rgb_d_max`; `null` where a column is constant or has fewer than 10 usable scenes). The progress log prints elapsed and last-scene time.

### Failure handling

- A scene that raises is recorded with `fatal_error` and the run continues.
- Out-of-memory errors empty the CUDA cache and continue.
- Any other CUDA error stops the run (exit code 1, `aborted` set in the summary), since a sticky CUDA error would invalidate every later scene.

### Diagnosis tool

`gsplat_mt/diagnose_mr4.py --scene-idx N --large` replays a large-scale scene with its sampled configuration. Besides per-pixel differences over exact (power-of-two) and inexact k, it reports how many 16x16 tiles the over-tolerance pixels fall in and the implied alpha of the Gaussian that would explain each difference (values near 1/255 point to gsplat's alpha cutoff).

### SLURM

`soc-cluster-large.sh`: one `h100-96` GPU, 3 hours, `--n-scenes 2000 --out results/large_scale.csv`.

## What was tested

2000 random scenes: 10-5000 Gaussians, 1-8 cameras, five resolutions (including a 120x90 size that is not a multiple of the 16-pixel tile), varied scene radius, scale, opacity, camera distance and field of view, across all 8 stress modes. Each scene went through MR1-MR5 and the three fault injections. The configuration sampler was also checked on its own: deterministic, in range, modes and resolutions drawn evenly, and no camera inside the scene radius.

## Findings

No scene crashed and the run did not abort. Running the study twice with the same seeds gave identical verdicts.

**Run time.** Per-scene time is about 16 ms regardless of scene size or mode (per-mode medians 15-17 ms; p99 0.12-0.15 s; max 0.18 s), so it is dominated by fixed overhead and not by the number of Gaussians or the resolution. The only large cost is the one-time CUDA extension compile, which falls on the first scene when the extension is not already built: 418 s in one run (about 8 minutes total), versus 0.8 s for the first scene (48 s total) in a run where it was already built.

| Relation | Result over 2000 scenes |
| --- | --- |
| MR1 packed vs unpacked | 1969 pass, 7 `isolated_discontinuity`, 24 `vacuous_pass`; max D_max 1.9e-03 |
| MR2 batched vs individual | 1992 pass, 8 `vacuous_pass`; D_max exactly 0 |
| MR3 covariance (correct) | 1976 pass, 24 `vacuous_pass`; D_max exactly 0 |
| MR4 world scale | 1888 pass, 85 `isolated_discontinuity`, 3 `violation`, 24 `vacuous_pass`; max D_max 5.5e-02 |
| MR5 permutation | 1960 pass, 16 `tie_ambiguous`, 24 `vacuous_pass` |

- **MR2 and MR3 stay bit-exact** at up to 5000 Gaussians, 8 cameras and 256x256.
- **MR5 holds away from depth ties.** 371 scenes had near-equal depths; in 16 of them the image differed (up to 1.8e-03). No scene without a depth tie differed.
- **MR1:** the 7 scenes over 1e-4 each had exactly one pixel over tolerance, all with 1300+ Gaussians, in `scale_huge`, `anisotropic` and `default` modes. The same alpha-cutoff effect found in week 8 is the likely cause, but it was not diagnosed for these.
- **Fault injections:** MR3 `xyzw_bug` and MR4 `no_translation` were detected in every non-empty scene; MR5 `misaligned` in every non-empty scene except the 264 `single` (N = 1) ones, where it cannot fire.
- **Empty renders:** the 24 `vacuous_pass` scenes are all `far_clipped` (8 of them empty from every camera).
- **Correlations (Spearman):** MR1 D_max rises with `overlap_frac` (0.79), `opaque_frac_cam0` (0.71), `coverage_cam0` (0.69) and N (0.53), and is near zero for camera count and opacity. Overlap, coverage and N are strongly confounded with each other, and most D_max values are 0 or noise, so this is weak evidence.

### MR4: tolerance failures grow with scene density

- 88 scenes had at least one pixel over 1e-4 at some k. The share of scenes doing so rises with N: 0.6% (N < 100), 2.6% (N 100-999), 15.6% (N >= 1000). More Gaussians per pixel means more chances to land on a rasterizer cutoff.
- 85 of them are `isolated_discontinuity`: typically one pixel, never more than 0.024% of the image.
- 3 scenes are `violation`, all at k = 0.01, all with 1400+ Gaussians (two `default`, one `anisotropic`), with 28 to 185 pixels over tolerance (0.21-0.28% of the image, about 10x the largest isolated case), D_max 1.8e-03 to 4.9e-03 and D_mae around 1e-6. The error is still sparse (over 99.7% of pixels match), but the gap to the isolated cases is clear.

## Open

- The three MR4 violations are not yet explained. Either they are the same cutoff flips at higher density (and the 0.05% isolated-pixel cutoff, chosen from a single week 8 case, is too tight), or something else. Replaying them with `diagnose_mr4.py --large` tests this: if power-of-two k gives exactly 0, the differences are rounding at a cutoff; if not, there is a real scale-dependent difference. A permanent version would add power-of-two factors to the MR4 sweep with exact zero tolerance.
