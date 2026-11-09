# Week 13: closing the coverage and detection gaps

The 100,000-scene study left two weaknesses (see `README_concluding_report.md`): most pixels in most scenes are background, so a `pass` rests on few pixels, and the planted MR5 fault went undetected in 12,440 scenes. Also, only MR2 rendered more than camera 0. This week changes the generator and the relations to address both and adds a tool to study the missed faults. Results are in the last section.

## Changes

### New scene type `dense_cover` (`gsplat_mt/scene.py`)

Fills the frame with several semi-transparent layers of Gaussians, so few pixels are plain background and many Gaussians contribute to each pixel.

- N is raised to at least 300. The scene radius, Gaussian scales and opacities are set from the cameras: `camera_hint` = (`dist_min`, `dist_max`, `fov_deg`, width / height), passed by `run_preliminary.build_scene_and_cameras`.
- The radius is the larger frame half-extent at the mid camera distance (times 1.15), capped at 0.8 x `dist_min` so no camera is inside the ball. Scales are set so N footprints of radius 2.5 scales total 4 layers over the ball's projected disc. Opacities are drawn from [0.05, 0.4].
- A standalone simulation (circular footprints, no gsplat) predicts 85-98% pixel coverage up to a 60 degree field of view, and 50-62% at 90 degrees with a wide camera-distance range, where the cap on the radius binds. The previous median coverage was 11%. These were predictions; the measured coverage is far lower (see Results).
- Opt-in: `DEFAULT_MODES` stays the original eight and is the runners' default, since adding a ninth would change every scene's sampled mode and break reproducibility of the earlier runs. `SCENE_MODES` is the eight plus `dense_cover`.
- In this mode N, the radius, the scale range and the opacity range come from the mode, not from the sampled `cfg_*` values, like the other modes that override fields.

### Every relation renders every camera

MR1, MR3, MR4 and MR5, and their planted-fault versions, now take a camera or a list and render all cameras in one batched call (`stack_cameras`, `as_camera_list` in `scene.py`). Before this, only MR2 used more than camera 0.

- MR4 scales the translation of every camera by k.
- `has_depth_tie` (MR5, and the MR4 `tie_ambiguous` relabel) is true if any camera has two Gaussians in front of it with depths closer than 1e-6.
- `compare_images` computes `n_px_over` per image and takes the largest, and `n_pixels` is H x W, so the isolated-pixel rule (at most 0.05% of pixels) stays per camera image.
- A `pass` becomes `vacuous_pass` only if every camera is empty. Rows carry `relations_cameras = all`; earlier CSVs rendered camera 0 only (MR2 aside), so their MR1/MR3/MR4/MR5 columns are not directly comparable.
- New columns: `coverage_min` (least-covered camera) and `n_covered_px` (covered pixels over all cameras), so a pass can be weighted by how many pixels were covered.
- `run_preliminary.build_scene_and_cameras` is now the one place scenes and cameras are built; `diagnose_mr4.py` uses it, so replays match. The `diagnose_*` tools still analyse camera 0 only.

### Study of undetected planted faults (`gsplat_mt/study_undetected_faults.py`)

Replays each scene of a `run_large_scale.py` CSV whose planted-fault verdict is `pass` (undetected on a non-empty image) and assigns a cause by experiment. `depth_tie` is recorded, so the causes can be counted with and without scenes that have near-equal depths. Checked in this order:

| Cause | Meaning | How it is tested |
| --- | --- | --- |
| `caught_by_other_cameras` | Detected when all cameras are rendered; camera 0 alone missed it | Rerun the fault on all cameras |
| `single_gaussian` | MR5: the shuffle is the identity | N = 1 |
| `visible_all_fixed_points` | MR5: every visible Gaussian keeps its index under the shuffle, so its color is unchanged | Permutation fixed points among Gaussians gsplat does not cull; `alt_detect_rate` is the share of 16 other shuffles that detect the fault |
| `visible_colors_unchanged` | MR5: visible Gaussians move, but the swapped colors are equal | Compare colors |
| `visible_isotropic` | MR3: every visible Gaussian has equal scales, so rotation is irrelevant | Scale ratio |
| `*_contribute_nothing` | The Gaussians the fault touches are visible to culling but never reach a pixel (below the 1/255 alpha cutoff, or fully occluded) | Set their opacity to 0 and check the render is unchanged |
| `unexplained` | None of the above | Manual look |

The MR4 fault has no classifier of its own: any undetected case is `unexplained` with its coverage recorded. In the 100,000-scene run MR3 and MR4 faults were detected in every non-empty scene, so only MR5 has data to study.

Hypothesis, not yet tested: the 77 non-single scenes where the shuffle fault was missed (75 with exactly one Gaussian visible from camera 0) are `visible_all_fixed_points`, i.e. the random shuffle left the one visible Gaussian in place. Nothing in the results records the shuffle, so this needed the replay.

Options: `--faults`, `--skip-modes` (single-Gaussian scenes are trivially undetectable for MR5), `--alt-seeds`, `--limit`, and the sampling arguments of the run being replayed (`--seed`, `--modes`, `--n-min`, `--n-max`, `--max-cameras`, `--world-scales`). If the sampled mode of a scene does not match the CSV's, the scene is skipped with a message, which catches mismatched arguments.

### SLURM

`soc-cluster-allcam.sh` (one `h100-96`, 3 hours):

1. `dense_cover` only, 500 scenes, seed 2000 (`results/dense_smoke.csv`): does the mode cover the frame? Compare `coverage_mean` with the other modes.
2. All nine modes, 20,000 scenes, seed 2000, every relation on every camera (`results/allcam_20000.csv`).
3. The study on that CSV, single-Gaussian scenes skipped (`results/undetected_allcam.csv`).
4. The study on the 100,000-scene CSV (`results/xl_scale.csv`) for the MR5 fault, single-Gaussian scenes skipped (`results/undetected_xl.csv`), with the default sampling arguments of that run.

The script has since been changed to the 100,000-scene run (the four steps above are commented out in the script, since their results exist); it has not been run yet, so the Results section describes the version above.

Step 4 replays scenes built with the new builder; for the original eight modes it builds the same scenes as before (`camera_hint` is only read by `dense_cover`), and the study renders them with all cameras, so it also shows whether the missed faults are caught once every camera is used.

## Not done

- The 100,000-scene run was not repeated with all cameras. Step 2 of the script ran 20,000 scenes (seed 2000, nine modes) instead, so its counts are not directly comparable with the 100,000-scene tables (different seed and size, and a ninth scene type). At 60 ms per scene with all cameras (19 ms before), 100,000 scenes would take about 100 minutes.
- The changes were not run locally (no GPU or PyTorch there); the cluster run was their first execution. It finished without errors or crashes.

## Results

Job 1102516: 21 minutes on one H100, no crashes, no fatal errors. Files: `results/dense_smoke.csv` (500 `dense_cover` scenes), `results/allcam_20000.csv` and the two `results/undetected_*.csv` studies.

### The 20,000 scenes (all cameras, nine modes)

| Relation | `pass` | `isolated_discontinuity` | `tie_ambiguous` | `violation` | `vacuous_pass` |
| --- | --- | --- | --- | --- | --- |
| MR1 packed vs unpacked | 19,549 | 392 | 0 | 0 | 59 |
| MR2 batched vs individual cameras | 19,941 | 0 | 0 | 0 | 59 |
| MR3 covariance vs scale+quat | 19,941 | 0 | 0 | 0 | 59 |
| MR4 exact (power-of-two k) | 19,941 | 0 | 0 | 0 | 59 |
| MR4 any k | 16,987 | 2,834 | 119 | 1 | 59 |
| MR5 permutation | 19,396 | 0 | 545 | 0 | 59 |

- **Exact relations stay exact with all cameras batched.** MR2, MR3 and the MR4 power-of-two check are exactly 0 in every non-empty scene.
- **Empty images:** 59 scenes (0.3%) are empty from every camera (54 `far_clipped`, 3 `wide_spread`, 2 `single`), against 1,424 of 100,000 (1.4%) empty from camera 0.
- **MR5:** 545 of the 6,047 tied non-empty scenes (9%) change the image, and none of the 13,894 untied ones. A scene counts as tied if any camera has near-equal depths, so 30% of scenes are tied, against 20% with camera 0 only.
- **MR4 `violation`:** 1 scene (`anisotropic`, 17 Gaussians, 5 cameras, no depth tie, 209 pixels over tolerance at worst, D_max 1.5e-3, worst k = 0.01). Not diagnosed; the diagnosis tools still look at camera 0 only.
- **Largest differences:** 3.8e-3 for MR1 and 3.9e-3 for MR4 `isolated_discontinuity` (the 1/255 cutoff bound); 0.056 for `tie_ambiguous`.
- **Planted faults:** MR3 and MR4 detected in every non-empty scene. MR5: undetected in 2,255 scenes, of which 2,250 are `single` (N = 1) and 5 are not.

### Findings per scene rose with the camera count, not because behavior changed

In the original eight modes the share of scenes with an MR4 `isolated_discontinuity` rose from 4.6% to 12.4%, and MR1 from 0.53% to 2.1%. By number of cameras per scene, new against old:

| Cameras | MR4 isolated, new | MR4 isolated, old | MR1 isolated, new | MR1 isolated, old |
| --- | --- | --- | --- | --- |
| 1 | 4.4% | 4.2% | 0.53% | 0.54% |
| 2 | 7.9% | 4.9% | 0.89% | 0.48% |
| 4 | 10.4% | 4.9% | 1.48% | 0.53% |
| 8 | 19.0% | 4.8% | 4.13% | 0.54% |

With one camera the rates equal the old ones. They then grow with the number of cameras, where the old rates (camera 0 only) stay flat. So the rise comes from checking more images per scene, and the relations behave as before. Each scene now has about 4.5 times as many images that can hit a rounding case, and no finding was bisected in this run. So "same mechanisms" is an inference from the rates, not a measurement.

### `dense_cover` and background pixels

Median coverage (share of pixels with alpha > 0, mean over cameras), per scene type: `scale_huge` 0.75, `dense_cover` 0.44, `far_clipped` 0.22, `wide_spread` 0.17, `default` 0.14, `anisotropic` 0.13, `opacity_extreme` 0.12, `scale_tiny` 0.04, `single` 0.006.

- `dense_cover` quartiles are 0.36, 0.44 and 0.54; none of its 2,240 scenes is under 10% and only 2% are above 80%. The median covered-pixel count is about 35,000 per scene.
- This is about half of the 85-98% the simulation predicted, and the shortfall does not depend on what was varied: median coverage is 0.41-0.46 across field-of-view buckets and 0.43-0.45 across N and camera-count buckets. It is lowest for square images (0.38) and higher for wide ones (0.55). Why the simulation overestimated was not found; its footprint model (circular footprints of radius 2.5 scales) is the first suspect.
- Opaque pixels (alpha > 0.99) are absent from it (median fraction 0.000), by design of the 0.05-0.4 opacities: `dense_cover` adds many partly covered pixels with several contributing layers, not saturated ones.
- Over all scenes the median coverage is 0.16 and 42.5% of non-empty scenes are under 10% coverage (47.5% in the 100,000-scene run); excluding `dense_cover` the share is 47.9%. The generator change helps only for the one scene type in nine.

### Undetected planted faults

| Run | Undetected (single-Gaussian scenes skipped) | Cause |
| --- | --- | --- |
| 100,000-scene run, camera 0 only | 77 of 86,213 | 58 `caught_by_other_cameras`, 18 `visible_all_fixed_points`, 1 `affected_gaussians_contribute_nothing` |
| 20,000-scene run, all cameras | 5 of 17,691 | 4 `visible_all_fixed_points`, 1 `affected_gaussians_contribute_nothing` |

- None has a depth tie, so ordering is not the cause in any of these scenes.
- **Camera 0 alone was the blind spot for 58 of the 77.** They are detected as soon as the other cameras are rendered (D_max from 0.003 to 0.84, median 0.34).
- **The fixed-point hypothesis holds for 22 scenes (18 + 4).** Every Gaussian visible in the cameras stayed at its index under the shuffle, so no color moved. Re-running with 16 other shuffles detects the fault on average in 92-95% of tries (per scene 81% to 100%, not always 100%, since a small N sometimes leaves the one visible Gaussian in place again). So these are a property of the shuffle, not of the relation.
- **2 scenes are `affected_gaussians_contribute_nothing`:** the Gaussians whose colors are swapped survive gsplat's culling, but zeroing their opacity leaves every pixel unchanged, so they never reach a pixel (below the 1/255 cutoff or fully occluded). Which of the two was not determined.
- MR3 and MR4 faults had no undetected non-empty scene in either run.
- The remaining miss rate is 0.03% of non-single scenes, against 0.09% before.

### Consequences for the earlier claims

- The concluding report's statements that MR1, MR3, MR4 and MR5 use camera 0 only, and its coverage and planted-fault numbers, describe the earlier run, not the current code.
- A shuffle fault that leaves the visible Gaussians in place could be avoided by drawing the shuffle without fixed points (a derangement), or by trying a second shuffle; not done.
