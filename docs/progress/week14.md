# Week 14: closing the all-camera open items

*Progress note. File paths are those of the current layout.*

Week 13 left four items open:

1. In violation scene 79993, one pixel (1.5e-4) stays over tolerance after removing the Gaussian that bisection kept.
2. Five violation scenes have one or two over-tolerance pixels in other cameras (seven in all), never bisected.
3. The all-camera run's 14,071 MR4 and 1,898 MR1 `isolated_discontinuity` scenes and 504 `tie_ambiguous` scenes were not bisected; "same mechanisms as the camera-0 runs" rested on the rates.
4. The planted MR5 fault still uses a random shuffle, which misses scenes where every visible Gaussian keeps its index.

## Changes

### Pattern of a bisection result

Each bisection now ends with a `PATTERN:` line, read off measured values:

| Pattern | Rule |
| --- | --- |
| depth-order flip | final transmittance unchanged, and a kept pair is one float32 step apart in depth |
| early stop | final transmittance changed, and it is between 1e-5 and 5e-4 (near the 1e-4 early-stop threshold) |
| cutoff flip | final transmittance changed, and exactly one kept Gaussian has alpha within 1% of 1/255 at the pixel |
| smooth difference | one kept Gaussian, nothing skipped or reordered |
| other | anything else; needs a manual look |

In week 11 the five MR4 cases where bisection kept 2 to 13 Gaussians were sorted by hand (one Gaussian at 1/255, the rest far from it); the cutoff-flip rule now makes that check. Each result also records the condition number of every kept Gaussian's conic (up to 50 Gaussians).

### Every over-tolerance pixel of the violation scenes (items 1 and 2)

`diagnose_allcam.py pixels` takes the nine violation scenes and every camera image with a pixel over tolerance, at every inexact factor for MR4 (0.01 and 100; the power-of-two factors are exactly 0). In each image it bisects the worst pixel not yet accounted for and counts every over-tolerance pixel that the kept Gaussians alone reproduce to at least half its difference (the bisection's own criterion) as accounted for. It repeats until every pixel is accounted for (at most 20 bisections per image). Then it removes every kept Gaussian and counts the pixels still over tolerance, and how many of those were not over tolerance in the full scene, since removing Gaussians can expose new differences. `sbatch soc-cluster-diagnose.sh violations` now runs it after `violations`.

### Bisection at the camera that shows the finding (item 3)

- `diagnose_mr4.py` and `diagnose_mr1.py` used to look at camera 0 only. With `--camera worst` (the default) they pick the camera whose image most plausibly made the verdict: the largest D_max if it is above the 4e-3 cutoff bound, else the most pixels over tolerance. `--camera 0` gives the old behaviour.
- With `--scenes-from`, a scene whose rgb image has nothing over tolerance is bisected on the alpha image instead (for MR4 also at the other inexact factor, 0.01 or 100), as was done by hand for the 52 alpha-only MR4 cases in week 11.
- `--out FILE` writes one JSON line per scene as it goes, `--resume` skips scenes already in the file, and `--summarize FILE...` prints the counts per pattern. `PART=j/K sbatch soc-cluster-diagnose.sh mr4 mr1` runs every K-th scene from the j-th, so K jobs cover all scenes.

### MR5 shuffle with no fixed points (item 4)

`run_mr5` now shuffles with a derangement drawn from the scene's seed: one cycle through all N Gaussians in a random order, so no Gaussian keeps its index (N = 1 cannot be shuffled). It applies to both the correct relation and the planted fault, which must share one shuffle. `random_shuffle` remains for comparison (`diagnose_allcam.py derangement` runs the fault with both). `study_undetected_faults.py` rebuilds the same derangement, and `diagnose_allcam.py nothing` takes `--shuffle random` for runs made before this change. With this change the MR5 columns of the earlier runs are no longer reproduced; MR1-MR4 are unaffected (checked in item 4 below).

## Runs

No log has an error.

- Items 1 and 2: `soc-cluster-diagnose.sh violations` (job 1145970, 3.5 minutes; `results/open_pixels.json`).
- Item 3: `PART=j/10 soc-cluster-diagnose.sh mr4 mr1` for j = 0..9 (jobs 1145971-1145980, 1.5 to 9 minutes each; `results/bisect/*.jsonl`), about 1,650 scenes per job.
- Item 4: the 100,000 scenes again (seed 2000, nine types) into `results/allcam_100000_derangement.csv`, keeping the earlier CSV for comparison, then `study_undetected_faults.py` on it (job 1145981, 106 minutes).

## Results

### Items 1 and 2: every over-tolerance pixel of the violation scenes

The nine violation scenes have 26 images (one camera at one inexact factor) with a pixel over tolerance in the rgb or the alpha image, 2,127 pixels in all. These counts include the alpha image, so some are higher than the rgb-only counts of week 13 (226 against 209 in scene 4896).

- Every pixel is accounted for, with one or two bisections per image: 1,726 pixels by a smooth difference, 397 by a cutoff flip, 4 by an early stop.
- Removing the kept Gaussians leaves 0 pixels over tolerance in all 26 images, and exposes no new ones.
- **Item 1:** in scene 79993 (camera 4, k = 0.01) the needle (kappa 2.6e4) accounts for 805 of the 806 pixels. The leftover pixel is a cutoff flip of a second, nearly round Gaussian (kappa 10).
- **Item 2:** every stray pixel is a cutoff flip of one Gaussian:

| Scene | Camera (k) | kappa of the kept Gaussian |
| --- | --- | --- |
| 42081 (MR1) | 4 | 1,672 |
| 34408 | 0 (0.01) | 27 |
| 42367 | 2 (0.01) | 4 |
| 42367 | 3 (0.01 and 100) | 5,423 |
| 77409 | 5 (100) | 2 |
| 79993 | 0 (0.01) | 242 |
| 79993 | 2 (0.01; 100) | 1,246; 12 |

- At the other inexact factor, the needle of each needle scene shows again in the same camera, as a smooth difference or a cutoff flip. Two images appear only there: scene 34408 camera 1 at k = 100 (5 pixels, smooth difference of another needle, kappa 7,521) and scene 77409 camera 2 at k = 0.01 (1 pixel, cutoff flip).

### Item 3: every all-camera finding bisected

All 16,473 findings, each at the camera that shows it. Counts per pattern, as the rule assigns them:

| Group | Cutoff flip | Depth-order flip | Early stop | Smooth difference | Other |
| --- | --- | --- | --- | --- | --- |
| MR4 `isolated_discontinuity` (14,071) | 13,592 (96.6%) | 270 (1.9%) | 201 (1.4%) | 7 | 1 |
| MR4 `tie_ambiguous` (504) | 71 (14.1%) | 413 (81.9%) | 16 (3.2%) | 2 | 2 |
| MR1 `isolated_discontinuity` (1,898) | 1,865 (98.3%) | 0 | 28 (1.5%) | 4 | 1 |

The camera-0 run (week 11) had 97.1%, 1.3%, 1.5% and 0.1% for MR4 `isolated_discontinuity`, 90% depth-order flips for `tie_ambiguous`, and 98.7% cutoff flips and 1.3% early stops for MR1.

- **Same mechanisms at about the same shares**, now measured rather than inferred from the rates.
- **The camera matters.** In 75% of the findings the camera that shows it is not camera 0. 129 findings (102 MR4, 27 MR1) have nothing over tolerance in that camera's rgb image and were bisected on its alpha image.
- **Kept sets.** Every depth-order flip keeps exactly 2 Gaussians, with a relative depth gap of at most 1.8e-7 (340 of the 683 are exactly tied in the base render). Every early stop keeps 3 to 375 Gaussians, with final T between 1.0e-4 and 1.3e-4. 36 cutoff flips keep more than one Gaussian, with exactly one at 1/255, like the five of week 11.
- **Needles.** All 13 smooth differences and the three "other" scenes are needles in `anisotropic` scenes (kappa 3.9e3 to 7.3e4). MR1 now has smooth differences among its `isolated_discontinuity` findings too (4); the camera-0 run had none.
- **The three "other" scenes (four findings), sorted by hand:**
  - 15446 (MR1, and MR4 at k = 0.01): a needle (kappa 2.0e4) whose alpha differs smoothly (0.262873 against 0.262745). Two Gaussians behind it, identical in both renders, are kept as well because they change how much the needle's alpha difference shows in the color. A smooth difference with three kept Gaussians, which the rule does not allow.
  - 16600 (MR4 `tie_ambiguous`): a needle (kappa 2.5e4) with alpha 0.4448 against 0.4442. A smooth difference whose relative change in T (1.07e-3) is just over the 1e-3 the rule uses for "transmittance changed".
  - 11686 (MR4 `tie_ambiguous`): a needle (kappa 7.3e4) with alpha 0.003872 and 0.003963 at the pixel, on either side of 1/255 (0.003922) but each more than 1% from it. A cutoff flip whose alpha moved by 2.3%.
- **`tie_ambiguous`:** 413 of the 504 are depth-order flips at the worst pixel. In the other 91 the worst pixel has another cause, so the near tie that relabeled the scene is not what bisection found there (as for 13 of 134 in the camera-0 run). In 36 of the 91, the image the CSV records (rgb at the worst factor) has at most 2 pixels over tolerance and D_max at most 4e-3, which alone would be `isolated_discontinuity`. So their violation is in an image the CSV does not record (an alpha image, or the other factor's), and that image was not bisected.

### Item 4: the 100,000 scenes with a derangement

- **MR1-MR4 are unchanged:** identical to `allcam_100000.csv` in all 100,000 rows (55 columns, 0 differences).
- **Planted MR5 fault:** undetected in 11,044 scenes, all `single` (one Gaussian cannot be shuffled). Detected in every non-empty scene with more than one Gaussian, including the 33 the random shuffle missed (27 `far_clipped`, 5 `wide_spread`, 1 `opacity_extreme`). `study_undetected_faults.py` has nothing left to study (0 scenes for each fault).
- **Correct MR5:** changes the image in 2,558 of the 29,955 tied scenes and in none of the 69,673 untied ones (2,583 with the random shuffle; a different shuffle reorders different tied pairs).
- 15,873 scenes have a finding (15,908 with the random shuffle). The 100,000 scenes took 100 minutes (102 before).

## Not done

- Bisection explains the worst pixel of each of the 16,473 findings; only the nine violation scenes were accounted for pixel by pixel.
- For the 91 `tie_ambiguous` scenes whose worst pixel is not a depth-order flip, the image that made each a violation was not located or bisected.
- Still open from week 13: why `dense_cover` covers about 45%; whether the three affected Gaussians that never reach a pixel are off the pixel grid or off screen; a labeling rule that uses the condition number; replaying depth-order flips in the kernel's exact arithmetic; double precision.
