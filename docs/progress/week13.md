# Week 13: closing the coverage and detection gaps

*Progress note. This week made every relation render every camera. Its 100,000-scene run (seed 2000, all nine scene types) is the one `soc-cluster.sh` runs, and its follow-up diagnoses are the steps of `soc-cluster-diagnose.sh`. File paths are those of the current layout.*

The 100,000-scene study left two weaknesses (see the concluding report, `docs/concluding_report.md`): most pixels in most scenes are background, so a `pass` rests on few pixels, and the planted MR5 fault went undetected in 12,440 scenes. Also, only MR2 rendered more than camera 0. This week changes the generator and the relations to address both and adds a tool to study the missed faults. Results are in the last section.

## Changes

### New scene type `dense_cover` (`gsplat_mt/scene.py`)

Fills the frame with several semi-transparent layers of Gaussians, so few pixels are plain background and many Gaussians contribute to each pixel.

- N is raised to at least 300. The scene radius, Gaussian scales and opacities are set from the cameras: `camera_hint` = (`dist_min`, `dist_max`, `fov_deg`, width / height), passed by `harness.build_scene_and_cameras`.
- The radius is the larger frame half-extent at the mid camera distance (times 1.15), capped at 0.8 x `dist_min` so no camera is inside the ball. Scales are set so N footprints of radius 2.5 scales total 4 layers over the ball's projected disc. Opacities are drawn from [0.05, 0.4].
- A standalone simulation (circular footprints, no gsplat) predicts 85-98% pixel coverage up to a 60 degree field of view, and 50-62% at 90 degrees with a wide camera-distance range, where the cap on the radius binds. The previous median coverage was 11%. These were predictions; the measured coverage is far lower (see Results).
- Opt-in at the time: the runners kept the original eight as their default, since adding a ninth changes every scene's sampled mode and so the scenes of the earlier runs. The runners now default to all nine (`SCENE_MODES`) with seed 2000; the eight are `ORIGINAL_MODES`.
- In this mode N, the radius, the scale range and the opacity range come from the mode, not from the sampled `cfg_*` values, like the other modes that override fields.

### Every relation renders every camera

MR1, MR3, MR4 and MR5, and their planted-fault versions, now take a camera or a list and render all cameras in one batched call (`stack_cameras`, `as_camera_list` in `scene.py`). Before this, only MR2 used more than camera 0.

- MR4 scales the translation of every camera by k.
- `has_depth_tie` (MR5, and the MR4 `tie_ambiguous` relabel) is true if any camera has two Gaussians in front of it with depths closer than 1e-6.
- `compare_images` computes `n_px_over` per image and takes the largest, and `n_pixels` is H x W, so the isolated-pixel rule (at most 0.05% of pixels) stays per camera image.
- A `pass` becomes `vacuous_pass` only if every camera is empty. Rows carry `relations_cameras = all`; earlier CSVs rendered camera 0 only (MR2 aside), so their MR1/MR3/MR4/MR5 columns are not directly comparable.
- New columns: `coverage_min` (least-covered camera) and `n_covered_px` (covered pixels over all cameras), so a pass can be weighted by how many pixels were covered.
- `build_scene_and_cameras` (now `gsplat_mt/harness.py`) is the one place scenes and cameras are built; `diagnose_mr4.py` uses it, so replays match. At the time the `diagnose_*` tools analysed camera 0 only; `diagnose_mr4.py` and `diagnose_mr1.py` now pick the camera that shows the finding (`--camera`).

### Study of undetected planted faults (`diagnosis/study_undetected_faults.py`)

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

Hypothesis (tested in the Results below): the 77 non-single scenes where the shuffle fault was missed (75 with exactly one Gaussian visible from camera 0) are `visible_all_fixed_points`, i.e. the random shuffle left the one visible Gaussian in place. Nothing in the results records the shuffle, so this needed the replay.

Options: `--faults`, `--skip-modes` (single-Gaussian scenes are trivially undetectable for MR5), `--alt-seeds`, `--limit`, and the sampling arguments of the run being replayed (`--seed`, `--modes`, `--n-min`, `--n-max`, `--max-cameras`, `--world-scales`). If the sampled mode of a scene does not match the CSV's, the scene is skipped with a message, which catches mismatched arguments.

### Runs

Job 1102516 (one `h100-96`, 3-hour limit):

1. `dense_cover` only, 500 scenes, seed 2000 (`results/dense_smoke.csv`): does the mode cover the frame? Compare `coverage_mean` with the other modes.
2. All nine modes, 20,000 scenes, seed 2000, every relation on every camera (`results/allcam_20000.csv`).
3. The study on that CSV, single-Gaussian scenes skipped (`results/undetected_allcam.csv`).
4. The study on the 100,000-scene CSV (`results/xl_scale.csv`) for the MR5 fault, single-Gaussian scenes skipped (`results/undetected_xl.csv`), with the default sampling arguments of that run.

Job 1102754 then ran 100,000 scenes with the same nine modes and seed 2000, and the study on its CSV, writing `results/allcam_100000.csv` and `results/undetected_allcam_100000.csv`. Its results are in the last Results section. These two steps are now `sbatch soc-cluster.sh large` and `sbatch soc-cluster-diagnose.sh undetected`.

Step 4 replays scenes built with the new builder; for the original eight modes it builds the same scenes as before (`camera_hint` is only read by `dense_cover`), and the study renders them with all cameras, so it also shows whether the missed faults are caught once every camera is used.

## Not done

- The first job (1102516) ran 20,000 scenes, not 100,000, so its counts are not directly comparable with the earlier camera-0 100,000-scene tables. The 100,000-scene all-camera run followed as job 1102754.
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
- **MR4 `violation`:** 1 scene (`anisotropic`, 17 Gaussians, 5 cameras, no depth tie, 209 pixels over tolerance at worst, D_max 1.5e-3, worst k = 0.01). Scene 4896; bisected later at the camera that shows it (camera 0), it is a cutoff flip of one needle-shaped Gaussian (see the 100,000-scene results).
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

### Consequences for the earlier claims (20,000-scene run)

- The concluding report's statements that MR1, MR3, MR4 and MR5 use camera 0 only, and its coverage and planted-fault numbers, describe the earlier run, not the current code.
- A shuffle fault that leaves the visible Gaussians in place could be avoided by drawing the shuffle without fixed points (a derangement), or by trying a second shuffle; a derangement detects all 33 scenes missed in the 100,000-scene run (Undetected planted faults, below).

## Results: 100,000 scenes, all cameras (job 1102754)

102 minutes on one H100 (60 ms per scene, median 48 ms), no crashes and no fatal errors. Nine modes, seed 2000, 1-8 cameras per scene (`results/allcam_100000.csv`). The first 20,000 scenes give verdicts identical to `allcam_20000.csv` in every relation (0 differences), so the sampler and the relations reproduce. The diagnoses in this section come from job 1141870 (`diagnosis/diagnose_allcam.py`, named `experiments_open_items.py` at the time; results in `results/open_*.json`; now `sbatch soc-cluster-diagnose.sh violations followup`).

| Relation | `pass` | `isolated_discontinuity` | `tie_ambiguous` | `violation` | `vacuous_pass` |
| --- | --- | --- | --- | --- | --- |
| MR1 packed vs unpacked | 97,729 | 1,898 | 0 | 1 | 372 |
| MR2 batched vs individual cameras | 99,628 | 0 | 0 | 0 | 372 |
| MR3 covariance vs scale+quat | 99,628 | 0 | 0 | 0 | 372 |
| MR4 exact (power-of-two k) | 99,628 | 0 | 0 | 0 | 372 |
| MR4 any k | 85,045 | 14,071 | 504 | 8 | 372 |
| MR5 permutation | 97,045 | 0 | 2,583 | 0 | 372 |

- **Exact relations:** MR2, MR3 and the MR4 power-of-two check are exactly 0 in all 99,628 non-empty scenes, with every camera rendered in one batch.
- **Violations (1 MR1, 8 MR4):** none points to a bug. Each was bisected at the camera that shows it, and each is one of the rounding patterns already seen in the camera-0 runs: 4 smooth differences, 2 cutoff flips and 3 early stops. Removing the Gaussians that bisection kept clears that camera's image in 8 of the 9. The MR1 one has 65 pixels over tolerance in one camera, by at most 3.3e-4: far below the 4e-3 size bound, but more than the 32 pixels (0.05% of 256x256) the isolated label allows. All 65 come from one needle-shaped Gaussian whose conic the packed and unpacked kernels round differently. Details in the next subsection.
- **Empty images:** 372 scenes (0.4%) are empty from every camera (342 `far_clipped`, 16 `wide_spread`, 13 `single`, 1 `opacity_extreme`).
- **MR5:** the permutation changes the image in 2,583 of the 29,955 tied scenes (8.6%) and in none of the 69,673 untied ones. 30% of non-empty scenes are tied (any camera), against 20% with camera 0 only.
- **Largest differences:** 3.9e-3 for MR1 and MR4 `isolated_discontinuity` (the cutoff bound); 0.28 for `tie_ambiguous`, 177 of the 504 above 4e-3.
- **Planted faults:** MR3 and MR4 faults detected in every non-empty scene (0 undetected). MR5: undetected in 11,077 scenes, 11,044 of them `single` (N = 1) and 33 not; a shuffle with no fixed points detects all 33 (see Undetected planted faults).

### Violations, bisected at the camera that shows them

`diagnose_allcam.py violations` replays each scene and compares every camera's image. It picks the camera that most plausibly made the verdict (the largest D_max if it is above the 4e-3 cutoff bound, else the most pixels over tolerance), bisects that image's worst pixel, then removes the kept Gaussians and counts what is still over tolerance in that image. kappa is the condition number of the kept Gaussian's conic: how strongly a rounding error in it is amplified (about the square of its length-to-width ratio; a round Gaussian has about 1).

| Relation | Scene | Mode | N | Cameras | Worst camera: pixels over tolerance, D_max | Kept Gaussians | kappa | Cause | Over tolerance after removing the kept Gaussians |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MR1 | 42081 | `anisotropic` | 1,740 | 6 | camera 3: 65 (limit 32), 3.3e-4 | 1 | 2.7e4 | smooth difference | 0 |
| MR4 | 4896 | `anisotropic` | 17 | 5 | camera 0: 209, 1.5e-3 | 1 | 2.0e4 | cutoff flip | 0 |
| MR4 | 34408 | `anisotropic` | 11 | 4 | camera 3: 45, 1.3e-4 | 1 | 2.0e4 | smooth difference | 0 |
| MR4 | 38581 | `opacity_extreme` | 108 | 7 | camera 2: 1, 4.7e-3 | 5 | 1.1 to 5.8 | early stop | 0 |
| MR4 | 42367 | `anisotropic` | 26 | 5 | camera 4: 136, 3.4e-3 | 1 | 1.3e4 | cutoff flip | 0 |
| MR4 | 69294 | `anisotropic` | 12 | 8 | camera 1: 37, 2.0e-4 | 1 | 1.9e4 | smooth difference | 0 |
| MR4 | 77409 | `opacity_extreme` | 120 | 7 | camera 2: 1, 1.5e-2 | 11 | 1.1 to 3.8 | early stop | 0 |
| MR4 | 79993 | `anisotropic` | 135 | 6 | camera 4: 806, 4.0e-4 | 1 | 2.6e4 | smooth difference | 1 pixel (1.5e-4) |
| MR4 | 94509 | `opacity_extreme` | 71 | 7 | camera 3: 1, 7.2e-2 | 14 | 1.0 to 3.9 | early stop | 0 |

- **The MR1 violation (scene 42081)** is the first in any run. The scene is fully covered (98.5% opaque pixels) and has 1,740 Gaussians. Bisection keeps one Gaussian, a needle about 165 times longer than wide. Nothing is skipped or reordered: the Gaussian is blended in both renders, but its alpha differs slightly along its length, and final transmittance at the worst pixel changes by only 0.07%. This is the smooth difference seen for MR4 in the camera-0 runs, now between packed and unpacked: the two projection kernels compute the conic with different rounding (week 11, root causes), and a kappa of 2.7e4 turns a last-digit difference into one of 3.3e-4. Removing that Gaussian leaves nothing over tolerance in camera 3, so it causes all 65 pixels. The scene has a near-tied depth pair in some camera, but MR1 is never relabeled for ties, and a single kept Gaussian cannot be an order flip.
- **The five MR4 needle scenes** behave the same way under scaling. In the two cutoff flips (209 and 136 pixels) the needle's alpha sits at 1/255 along its edge, so one rounding difference flips many pixels at once. In the three smooth differences its alpha differs slightly everywhere.
- **The three `opacity_extreme` scenes** are early stops with round Gaussians (kappa close to 1) of opacity 0.999 or 1: final transmittance at the pixel crosses 1e-4 in one render and not the other. They exceed the 4e-3 bound because the pixel is nearly opaque.
- **Why they are `violation`s:** the six needle cases fail the pixel-count rule, which one needle can exceed; the three early stops fail the size bound. A rule that also uses kappa or the number of Gaussians involved could separate these from violations that matter; not tried.
- **Not yet explained:** in scene 79993 one pixel (1.5e-4) remains after removing the kept Gaussian. Five scenes have stray pixels in other cameras that were not bisected: 42081 (camera 4, 1.3e-4), 34408 (camera 0, 1.8e-3), 42367 (cameras 2 and 3, 2.0e-3 and 2.5e-3), 77409 (camera 5, 4.2e-4), 79993 (cameras 0 and 2, 2.7e-3 and 1.3e-4). Week 14 bisected every over-tolerance pixel of these scenes: the leftover pixel of 79993 is a cutoff flip of a second, nearly round Gaussian (kappa 10), and every stray pixel is a cutoff flip of one Gaussian.
- The first scene (4896) is the one already listed in the 20,000-scene run.

### Rates by camera count (original eight modes, 88,940 scenes)

| Cameras | MR4 isolated | MR1 isolated | Scenes |
| --- | --- | --- | --- |
| 1 | 4.7% | 0.56% | 10,992 |
| 2 | 8.0% | 1.05% | 11,150 |
| 4 | 11.6% | 1.69% | 11,109 |
| 8 | 18.5% | 3.41% | 11,168 |

Overall 12.5% (MR4) and 2.0% (MR1), against 4.6% and 0.53% with camera 0 only. The 20,000-scene table already showed the effect; the rates with one camera again equal the old ones, and the 100,000 scenes put the increase beyond sampling noise. The `isolated_discontinuity` and `tie_ambiguous` scenes of this run were not bisected, so "the same mechanisms as before" rests on these rates. Week 14 bisected all of them, each at the camera that shows it: 97% of MR4 and 98% of MR1 `isolated_discontinuity` are cutoff flips, and 82% of `tie_ambiguous` are depth-order flips, close to the camera-0 run's shares.

### Coverage

Median coverage by scene type: `scale_huge` 0.74, `dense_cover` 0.45, `far_clipped` 0.21, `wide_spread` 0.18, `default` 0.13, `anisotropic` 0.12, `opacity_extreme` 0.12, `scale_tiny` 0.04, `single` 0.006. `dense_cover` quartiles are 0.36, 0.45, 0.55; none of its 11,060 scenes is under 10% coverage and 2% are above 80%, as in the 20,000-scene run. Over all non-empty scenes the median is 0.16, and 42.5% are under 10% coverage (47.8% excluding `dense_cover`; 47.5% in the camera-0 run). The median is about 9,400 covered pixels per scene over all cameras. The mode adds covered pixels to one scene type in nine and leaves the overall share of sparse scenes nearly unchanged.

Why `dense_cover` covers about 45% and not the predicted 85-98% (`diagnose_allcam.py coverage`, 60 of its scenes): the replayed coverage matches the CSV exactly, and 100% of the Gaussians survive culling in at least one camera, so culling does not limit coverage. Scaling every Gaussian by 1, 2 and 3 gives median coverage 0.43, 0.59 and 0.74.

- Small scenes (300 Gaussians) respond strongly to the scaling (up to 70 to 94% at three times the size), so footprint size limits them.
- Large scenes (over 2,000 Gaussians) hardly respond (for example 0.35 to 0.46, 0.28 to 0.34), so for them the region covered by all the Gaussians together is what limits coverage.
- The median ratio of measured screen radius to the radius assumed by the simulation is 1.38. This is a rough check only. Why the simulation overestimated coverage is still not found.

### Undetected planted faults (single-Gaussian scenes skipped)

33 of 88,584 non-empty scenes (0.037%), against 77 of 86,213 (0.09%) with camera 0 only. None has a depth tie. MR3 and MR4 planted faults were never undetected.

| Cause | Scenes | Detected with a derangement |
| --- | --- | --- |
| `visible_all_fixed_points` | 30 | 30 |
| `affected_gaussians_contribute_nothing` | 3 | 3 |

- **Fixed points (30 scenes):** every visible Gaussian stayed at its index under the random shuffle, so no visible color moved. Re-running with 16 other random shuffles detects the fault in 92% of tries on average. The 33 include the 5 undetected scenes of the 20,000-scene run, since its scenes are the first 20,000 here.
- **Contribute nothing (3 scenes):** the one affected Gaussian that survives culling (the renderer's early step that discards Gaussians that cannot appear in a camera) never reaches a pixel even when rendered alone (`diagnose_allcam.py nothing`), with opacities 0.61, 0.15 and 0.61, well above the 1/255 cutoff. So the cause is its footprint, not occlusion or low opacity: it lies off the pixel grid or off screen, which the experiment cannot tell apart. Every Gaussian that does reach a pixel stayed in place, so these are fixed-point cases too.
- **Derangement (`diagnose_allcam.py derangement`):** a shuffle that moves every Gaussian (no Gaussian keeps its index) detects the planted fault in all 33, and the correct relation still passes under it in all 33, so there is no false alarm. Tested on these 33 scenes only; the full 100,000-scene run was not repeated with a derangement. Week 14 repeated it: the fault is then detected in every non-empty scene with more than one Gaussian, and MR1-MR4 are identical in every scene.
