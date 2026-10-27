# Week 11: completing the Week 10 follow-ups

The two proposed oracle changes, and tools to settle the three open diagnosis questions. Results below are filled in only for what has been run.

## Changes

### Oracle (`gsplat_mt/metrics.py`, `gsplat_mt/run_preliminary.py`)

- **Exact-zero MR4 check.** `mr4_exact_verdict` covers the power-of-two factors in the sweep, where scaling is exact in binary: the two renders must be identical (tolerance 0, no `isolated_discontinuity`, no `tie_ambiguous`). Any difference there is reported as `violation`; the largest is recorded as `mr4_exact_d_max`. `mr4_verdict` still covers every factor with the 1e-4 tolerance. The default sweep is now `0.01 0.0078125 0.25 4 100 128` (adds 1/128 and 128; the fault injection runs at the largest factor, now 128). In the four scenes replayed in Week 10 every power-of-two factor was exactly 0; whether that holds in general is what the exact check now measures.
- **Size bound on `isolated_discontinuity`.** The label now also requires D_max of at most 4e-3 (`ISOLATED_MAX_D_MAX`): one Gaussian with alpha near 1/255 blended in one render and skipped in the other changes a pixel by at most color x alpha x T, about 3.9e-3. A single-pixel difference above that is a `violation`. Applies to MR1 and MR4.
- `is_power_of_two(k)` in `metrics.py`. The new verdict is part of `run_large_scale.py`'s findings and the verdict tables.

### Diagnosis tools

- `gsplat_mt/diagnose_mr4.py`
  - `--bisect` ends with a `CLASS:` line read off measured values: final transmittance changed (a Gaussian was blended in one render and skipped in the other; flagged when T is near the 1e-4 early-stop threshold) or unchanged with a kept pair at a depth gap of about one float32 step (blend order flipped), or unclassified.
  - `--by T` bisects the pixel with the largest relative change in final transmittance instead of the largest color difference (aimed at the early-stop pattern).
  - `--scenes-from CSV --where column=value` bisects every matching scene at its `mr4_worst_k` and prints a count per class.
- `gsplat_mt/diagnose_mr1.py`: the same bisection for packed vs unpacked (MR1), reading depths and alphas from each path's own `meta`, packed or not.
- `gsplat_mt/probe_ties.py`: two overlapping Gaussians with equal depth, or one float32 step apart, rendered in both storage orders and with `packed` on and off. It prints which Gaussian ends up in front, so whether ties go by depth, by index or by neither is observed.
- `soc-cluster-diagnose-further.sh`: reruns the 2000-scene study with the new oracle, then runs the probe and all bisections below on the new results. `soc-cluster-xl.sh` (100,000 scenes) also picks up the new oracle.

## What the run answers

| Open item from Week 10 | Tool |
| --- | --- |
| MR1 single-pixel cases | `diagnose_mr1.py` on every MR1 `isolated_discontinuity` scene |
| Which of the MR4 isolated cases are cutoff flips and which depth swaps | `diagnose_mr4.py --scenes-from` over every remaining `isolated_discontinuity` and every `tie_ambiguous` scene |
| How gsplat resolves equal depths | `probe_ties.py` |
| Early-stop pattern in one violator | `diagnose_mr4.py --bisect --by T` on the three former violators |
| Exact-zero claim for power-of-two k | `mr4_exact_verdict` over the 2000 scenes (and the 100,000-scene run) |

## Read from gsplat 1.5.3 source (`IntersectTile.cu`, `RasterizeToPixels3DGSFwd.cu`)

- Each Gaussian-tile pair gets a 64-bit key: image id | tile id | the depth's 32 float bits, reinterpreted as an integer (no quantization), and the pairs are sorted by a CUB radix sort, which is stable. So two Gaussians with exactly equal depth in a tile keep the order in which their pairs were generated, and depths one float32 step apart are always ordered by depth. That generation order is not read from the source here; `probe_ties.py` tests whether it follows the Gaussian index.
- The early stop is exclusive: when `next_T = T * (1 - alpha)` is at most 1e-4, the pixel stops and that Gaussian is not blended. So a pixel whose alpha rounds slightly differently can skip a Gaussian with a large alpha (a large change in T) while changing the color by at most about `color x alpha x T` with T below about 1e-4 / (1 - alpha).

## Results

Rerun of the 2000 scenes plus the diagnosis tools on the cluster (one H100; 53 s for the study with the extension already built).

### Oracle

- **Exact check:** over all 2000 scenes `mr4_exact_d_max` is exactly 0 (1976 `pass`, 24 `vacuous_pass`, no violation), over the power-of-two factors 1/128, 1/4, 4 and 128. gsplat's world-scale handling is bit-exact there, in every scene tried and not just the four replayed in Week 10.
- **Unchanged elsewhere:** MR1, MR2, MR3, MR4 (before the tie relabel), MR5 and fault-injection verdicts match the previous run scene for scene; the fault injections are still detected in every non-empty scene.
- **Size bound:** all 30 untied isolated cases have a single-Gaussian difference of at most 3.9e-3, so none was reclassified. The three violators (28-185 pixels) and the 5.5e-02 outlier (D_max 5.5e-02 over 5 pixels) are the only scenes outside the isolated rule.

### How gsplat orders equal depths (`probe_ties.py`)

- Equal depth: the Gaussian stored first (lower index) is drawn in front, in both `packed=False` and `packed=True`; swapping the storage order swaps the image.
- One float32 step apart (either direction): the nearer one is in front regardless of storage order, in both modes.
- So exactly tied depths are broken by index, and a permutation (MR5) can change the image for a tied pair. Why MR5 reported 0 on the violators was not tested; one random permutation flips a given pair about half the time.

### MR4 isolated cases (30, `diagnose_mr4.py --scenes-from`)

All 30 are the same cutoff flip: one Gaussian out of 1-4346 is enough to reproduce the difference, its alpha at the pixel center is 0.003922 (1/255 = 0.003922) in both renders (the last printed digits differ in two scenes: 0.003921 vs 0.003922 and 0.003923 vs 0.003918), and final transmittance changes by 3.9e-03 (= 1/(1 - 1/255) - 1). Both k = 0.01 and k = 100 occur; the set includes an N = 1 scene and an N = 13 scene, and seven of the eight stress modes (all but `opacity_extreme`).

### MR4 tied scenes (58, the former `tie_ambiguous`)

- 54 are the identical single-Gaussian cutoff flip (alpha 0.003922, transmittance change 3.9e-03). Their scene contains a near-equal-depth pair somewhere, but the pixel's difference does not involve it, so the scene-level tie flag relabeled cutoff flips and says nothing about the cause of a given pixel. It now applies to `violation` only; on these 2000 scenes that leaves 84 `isolated_discontinuity`, 4 `tie_ambiguous` and 0 `violation` (computed from the run's pixel counts and D_max, not rerun).
- 4 are depth-order flips: the three violators and the 5.5e-02 outlier. Two Gaussians reproduce the pixel, final transmittance changes by at most 4.7e-06, and the pair's relative depth gaps are 6.0e-08, 0, 9.9e-08 and 0.
- 1 is neither (an `opacity_extreme` scene): 17 Gaussians are needed, transmittance changes by 86%, final T is near 1e-4, and the subset's pixel difference is 5.9e-04. That fits the early stop but was not examined further.

### Early-stop pattern in the former violators (`--by T`)

- One violator has a pixel whose final T is 1.2e-04 in one render and 5.0e-04 in the other (77%, the pattern seen in Week 10), right at the early-stop threshold. 12 Gaussians reproduce more than half of it, and they include the tied pair. So that pixel combines the order flip and the early stop; that the flip is what pushes T across 1e-4 is consistent with this but not isolated.
- For another, the pixel chosen had only a 2.7e-05 change in T, so it is not an early-stop pixel, and the bisection stopped at its 20,000-render cap with 1716 Gaussians left: unresolved.
- For the third, no pixel over tolerance has any change in final T (among pixels with T above 1e-4), which fits a pure order flip. (The tool picked an unrelated pixel here; fixed to report "no over-tolerance pixel changed its final transmittance".)

### MR1 single-pixel cases (7, `diagnose_mr1.py`)

All 7 are the same cutoff flip between packed and unpacked: one Gaussian reproduces it, alpha at the pixel center is 0.003922 in both (0.003921 in one render for one case), transmittance changes by 3.9e-03, subset difference 5.9e-04 to 3.7e-03. The scenes have 1310-4854 Gaussians (`scale_huge` 3, `anisotropic` 3, `default` 1). It was not tested why the two code paths land on opposite sides of 1/255 in the last digits.

### Not done

- The 100,000-scene run (`soc-cluster-xl.sh`) has not been run with the new oracle.
- The tie relabel narrowing was applied after this run and has not been run on the cluster.
