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

`run_mr5` now shuffles with a derangement drawn from the scene's seed: one cycle through all N Gaussians in a random order, so no Gaussian keeps its index (N = 1 cannot be shuffled). It applies to both the correct relation and the planted fault, which must share one shuffle. `random_shuffle` remains for comparison (`diagnose_allcam.py derangement` runs the fault with both). `study_undetected_faults.py` rebuilds the same derangement, and `diagnose_allcam.py nothing` takes `--shuffle random` for runs made before this change. With this change the MR5 columns of the earlier runs are no longer reproduced; MR1-MR4 are unaffected.

## Runs

- Items 1 and 2: `soc-cluster-diagnose.sh violations`.
- Item 3: `PART=j/10 soc-cluster-diagnose.sh mr4 mr1` for j = 0..9, about 1,650 scenes per job (about 1.7 s per scene in week 11).
- Item 4: the 100,000 scenes again (seed 2000, nine types) into `results/allcam_100000_derangement.csv`, keeping the earlier CSV for comparison, then `study_undetected_faults.py` on it.

## Results

Pending.

## Not done

- The code above is checked only statically (imports, argument parsing, the pattern rule on made-up values); none of it has run yet.
