# Week 11: closing the Week 10 open items

Week 10 left five items open: the oracle (exact-zero check on power-of-two factors, size bound on `isolated_discontinuity`), the MR1 single-pixel cases, the MR4 isolated cases, how gsplat orders equal depths, and the early-stop pattern in one violator. This week applies the two oracle changes and adds a diagnosis tool for each remaining item; the results follow.

## Changes

### Oracle (`gsplat_mt/metrics.py`, `gsplat_mt/run_preliminary.py`)

- **Exact-zero MR4 check.** `mr4_exact_verdict` covers the power-of-two factors in the sweep, where scaling is exact in binary: the two renders must be identical (tolerance 0, no `isolated_discontinuity`, no `tie_ambiguous`). Any difference there is reported as `violation`; the largest is recorded as `mr4_exact_d_max`. `mr4_verdict` still covers every factor with the 1e-4 tolerance. The default sweep is now `0.01 0.0078125 0.25 4 100 128` (adds 1/128 and 128; the fault injection runs at the largest factor, now 128). In the four scenes replayed in Week 10 every power-of-two factor was exactly 0; whether that holds in general is what the exact check now measures.
- **Size bound on `isolated_discontinuity`.** The label now also requires D_max of at most 4e-3 (`ISOLATED_MAX_D_MAX`): one Gaussian with alpha near 1/255 blended in one render and skipped in the other changes a pixel by at most color x alpha x T, about 3.9e-3. A single-pixel difference above that is a `violation`. Applies to MR1 and MR4.
- `is_power_of_two(k)` in `metrics.py`. The new verdict is part of `run_large_scale.py`'s findings and the verdict tables.

### Diagnosis tools

- `gsplat_mt/diagnose_mr4.py`
  - `--bisect` ends with a `CLASS:` line read off measured values: final transmittance changed (a Gaussian was blended in one render and skipped in the other; flagged when T is near the 1e-4 early-stop threshold) or unchanged with a kept pair at a depth gap of about one float32 step (blend order flipped), or unclassified. One float32 step is one ULP (unit in the last place), the gap between a float and the next representable float, about 1.2e-7 relative.
  - `--by T` bisects the pixel with the largest relative change in final transmittance instead of the largest color difference (aimed at the early-stop pattern).
  - `--scenes-from CSV --where column=value` bisects every matching scene at its `mr4_worst_k` and prints a count per class.
- `gsplat_mt/diagnose_mr1.py`: the same bisection for packed vs unpacked (MR1), reading depths and alphas from each path's own `meta`, packed or not.
- `gsplat_mt/probe_ties.py`: two overlapping Gaussians with equal depth, or one float32 step apart, rendered in both storage orders and with `packed` on and off. It prints which Gaussian ends up in front, so whether ties go by depth, by index or by neither is observed.
- `soc-cluster-diagnose-further.sh`: reruns the 2000-scene study with the new oracle, then runs the probe and all bisections below on the new results. `soc-cluster-xl.sh` (100,000 scenes) also picks up the new oracle.

## What the run answers

| Open item from Week 10 | Tool |
| --- | --- |
| Exact-zero claim for power-of-two k | `mr4_exact_verdict` over the 2000 scenes (and the 100,000-scene run) |
| MR1 single-pixel cases | `diagnose_mr1.py` on every MR1 `isolated_discontinuity` scene |
| Which of the MR4 isolated cases are cutoff flips and which depth swaps | `diagnose_mr4.py --scenes-from` over every remaining `isolated_discontinuity` and every `tie_ambiguous` scene |
| How gsplat resolves equal depths | `probe_ties.py` |
| Early-stop pattern in one violator | `diagnose_mr4.py --bisect --by T` on the three former violators |

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
- For the third, no pixel over tolerance has any change in final T (among pixels with T above 1e-4), which fits a pure order flip.

### MR1 single-pixel cases (7, `diagnose_mr1.py`)

All 7 are the same cutoff flip between packed and unpacked: one Gaussian reproduces it, alpha at the pixel center is 0.003922 in both (0.003921 in one render for one case), transmittance changes by 3.9e-03, subset difference 5.9e-04 to 3.7e-03. The scenes have 1310-4854 Gaussians (`scale_huge` 3, `anisotropic` 3, `default` 1). It was not tested why the two code paths land on opposite sides of 1/255 in the last digits.

### Extra-large run (100,000 scenes, `soc-cluster-xl.sh`)

Same sampler and seed as the 2000-scene study, with the narrowed tie relabel. 43 minutes (median 19 ms per scene); no scene crashed and the run did not abort. The first 2000 scenes reproduce the 2000-scene results exactly under the narrowed relabel: MR4 84 `isolated_discontinuity`, 4 `tie_ambiguous`, 0 `violation`.

| Relation | Result over 100,000 scenes |
| --- | --- |
| MR1 packed vs unpacked | 98,044 pass, 532 `isolated_discontinuity`, 1,424 `vacuous_pass`; max D_max 3.8e-03 |
| MR2 batched vs individual | 99,559 pass, 441 `vacuous_pass`; D_max exactly 0 |
| MR3 covariance (correct) | 98,576 pass, 1,424 `vacuous_pass`; D_max exactly 0 |
| MR4 exact (power-of-two k) | 98,576 pass, 1,424 `vacuous_pass`; D_max exactly 0 |
| MR4 world scale | 93,794 pass, 4,643 `isolated_discontinuity`, 134 `tie_ambiguous`, 5 `violation`, 1,424 `vacuous_pass`; max D_max 0.15 |
| MR5 permutation | 97,807 pass, 769 `tie_ambiguous`, 1,424 `vacuous_pass`; no `violation` |

- **Bit-exact relations:** MR2, MR3 and the MR4 power-of-two check are exactly 0 in every non-empty scene, up to 5000 Gaussians, 8 cameras, 256x256.
- **MR5:** 19,748 of the 98,576 non-empty scenes (20%) have near-equal depths, 15,955 of them with 1000+ Gaussians. The permutation changes the image in 769 of the tied scenes (3.9%) and in none of the 78,828 untied scenes.
- **MR1:** no scene is over the 4e-03 bound; the 532 isolated cases have at most 4 pixels over tolerance (495 have 1), include an N = 1 scene, and span `scale_huge` (256), `anisotropic` (122), `default` (72), `opacity_extreme` (44), `wide_spread` (21), `far_clipped` (16) and `single` (1).
- **MR4 isolated:** 4,643 scenes, largest D_max 3.92e-03 (the cutoff bound), never more than 0.049% of the pixels.
- **MR4 `tie_ambiguous`:** 134 scenes, all tied, 111 with 1000+ Gaussians; 49 have D_max above 4e-03. These are violations by pixel count or size inside a tied scene, not individually diagnosed in this run.
- **Fault injections:** MR3 `xyzw_bug` and MR4 `no_translation` are detected in every non-empty scene. MR5 `misaligned` did not fire in 12,440 scenes (12.4%), about the share of `single` (N = 1) scenes, where the permutation is the identity.
- **Correlations (Spearman):** MR1 D_max with overlap 0.79, opaque fraction 0.70, coverage 0.69, N 0.49; MR4 D_max with N 0.45, mean scale -0.42, overlap -0.35. Same pattern and caveats as in Week 10.

**Five MR4 violations in scenes with no depth tie.** MR5 passes on all five, so the tie flag does not explain them.

| Mode | N | Pixels over tolerance | D_max | Worst k |
| --- | --- | --- | --- | --- |
| `anisotropic` | 17 | 9 | 1.9e-03 | 100 |
| `opacity_extreme` | 173 | 1 | 7.1e-03 | 100 |
| `anisotropic` | 27 | 50 | 2.3e-04 | 0.01 |
| `anisotropic` | 912 | 280 | 2.0e-03 | 100 |
| `opacity_extreme` | 371 | 2 | 9.3e-03 | 100 |

Two are single or double pixels above the 4e-03 bound (so not a lone cutoff flip), and two have 50 and 280 pixels with a small D_max. Their cause is not known. Violations occur at about 0.006% of untied scenes (5 of 78,828) and 0.68% of tied scenes (134 of 19,748), about 100 times higher, so ties account for most MR4 violations; but five untied cases show that a tie is not the only cause, and some of the 134 tied ones may not involve a tie either (the tie flag is per scene, and 54 of the 58 tied scenes in the 2000-scene run were plain cutoff flips).

### Bisection of the 100,000-scene findings (`soc-cluster-diagnose-xl.sh`)

Same `CLASS:` rule as above, on the five untied violations, every second `tie_ambiguous` scene (67 of 134), every 40th MR4 `isolated_discontinuity` scene (117 of 4,643) and every 10th MR1 `isolated_discontinuity` scene (54 of 532). Only samples were run; the counts below are of those samples.

| Group (sampled) | Single-Gaussian cutoff flip | Depth-order flip | Early stop | Other |
| --- | --- | --- | --- | --- |
| MR4 untied violations (5 of 5) | 2 | 0 | 2 | 1 |
| MR4 `tie_ambiguous` (67) | 5 | 60 | 2 | 0 |
| MR4 `isolated_discontinuity` (117) | 115 | 1 | 1 | 0 |
| MR1 `isolated_discontinuity` (54) | 52 | 0 | 0 | 2 not bisected (see below) |

- **Depth-order flips are the tied violations.** In 60 of 67 sampled tied scenes two Gaussians reproduce the pixel, final transmittance changes by at most 6.2e-06, and their relative depth gap is at most 1.6e-07 (0 in 25 of them). D_max of the pair alone reaches 0.21.
- **Early-stop flips can exceed the cutoff bound.** Two untied violations (both `opacity_extreme`, D_max 7.1e-03 and 9.3e-03) need 20 and 14 Gaussians; in the first, final T is 1.0002e-04 in one render and 7.7e-03 in the other, so a Gaussian with alpha near 0.99 was blended in one render and not in the other at the 1e-4 early-stop threshold. Skipping it changes the color by up to color x alpha x T with T up to 1e-4 / (1 - alpha), about 0.1 at gsplat's 0.999 alpha clamp, so the 4e-03 bound does not cover this mechanism. The same pattern appears in 2 sampled tied scenes (one with D_max 7.7e-02) and 1 sampled isolated scene (D_max 2.4e-04).
- **Some violations are not flips of any kind.** Two untied `anisotropic` violations (N = 912 with 280 pixels over tolerance, N = 27 with 50 pixels) and one more (N = 17) look like a cutoff flip or like no flip at their worst pixel; the follow-up diagnoses below show what they are.
- **MR1:** 52 of 54 sampled cases are the same single-Gaussian cutoff flip as before. For 2, no rgb pixel is over tolerance (the over-tolerance image is alpha); those are covered below.
- **Size bound is exact for cutoff flips, not for the other mechanisms.** The sampled single-Gaussian flips all have a subset difference of at most 3.9e-03.

### Follow-up diagnoses (`soc-cluster-diagnose-open.sh`)

**Early stop confirmed (two untied `opacity_extreme` violations).** Replaying the blending at the worst pixel in float32 from each render's own values shows the stop. In the first scene, the Gaussian with alpha 0.987 gives `next_T` = 1.000002e-04 in the base render (above the threshold, blended) and 9.99998e-05 in the scaled render (below it, so the pixel stops without blending it); the kernel's final T is 1.0e-04 versus 7.7e-03. In the second, the replay puts `next_T` just below 1e-4 in both renders (9.99989e-05 and 9.99984e-05), but the kernel blended it in the base render (final T 1.0e-04 against 1.0e-02 in the scaled render), so there the torch replay, which evaluates the alpha expression differently from the kernel, is within its own error of the threshold and does not reproduce the kernel's decision (the kernel-arithmetic replay under Root causes does). The kernel final-T values come from `1 - alpha` of a float32 alpha image and are quantized to about 6e-08.

**The 280-pixel violation is a conic difference, not a boundary flip.** (A conic is the three numbers describing a Gaussian's elliptical 2D footprint on the screen; alpha at a pixel is computed from it.) Removing its one Gaussian leaves 0 pixels over tolerance (D_max 1.8e-06), so that Gaussian causes all 280. But at the 280 pixels its alpha is 0.5 at most and within 1% of 1/255 at only 2 of them (min 0.0039, max 0.53). At the worst pixel, alpha is 0.003906 in one render and 0.003928 in the other, a 0.56% difference. This is the pattern of the smooth difference below, with one pixel that also happens to cross 1/255. (The Gaussian's axis ratio, the length of the footprint's long axis divided by its short axis, was not measured.)

**The N = 17 violation comes from the alpha image.** At k = 100 the rgb image has 9 pixels over tolerance (limit 32) but the alpha image has 47, so it fails the pixel-count rule. At k = 0.01 both images pass the rule (24 and 4 pixels). Every other factor, including all power-of-two ones, is exactly 0.

**The smooth difference (N = 27, `anisotropic`, k = 0.01).** One Gaussian whose projected 2D shape (footprint) is a thin needle, with an axis ratio of about 103. Its conic differs between the base and the scaled render by 24,610 ULP (0.911904 vs 0.910438 for the first entry; at this magnitude one ULP is about 6e-8, and two calculations of the same value normally differ by 0 to 3 ULP); `means2d`, the Gaussian's center on the screen, differs by 1 ULP. A float64 reference projection (a re-implementation, written for this project, of gsplat's EWA projection, the step that turns each 3D Gaussian into its 2D footprint) gives the same conic for both renders, as scale invariance requires (scaling the whole world by k must not change the footprint on screen), and the two float32 renders are 1.2e-03 (base) and 3.9e-04 (scaled) away from it, against about 1e-07 for the means. Because the true value is scale-invariant, at least one render must be off by at least 8e-04 whatever the reference formula, and the alpha at the pixel differs accordingly (0.306468 vs 0.306767; reference 0.306828 for both). The size of the error fits a covariance with a condition number (how much a relative input error is amplified in the output) around 1e4 (axis ratio squared) and float32 relative precision of about 6e-08, which is a consistent explanation; the root-cause section below measures how this scales with the conditioning of the covariance.

**MR1 packed vs unpacked: the projection outputs already differ.** In all 59 bisected MR1 cutoff-flip scenes (7 from the 2000-scene study, 52 from the 100,000-scene sample), the kept Gaussian's conic differs between the paths, by 1 to 7 ULP in 51 of them and by 12 to 10,592 ULP in the other 8, while `means2d`, `opacities`, `depths` and `radii` are bit-identical. Over every Gaussian visible in both paths, `means2d`, `opacities`, `depths` and `radii` never differ (0 of 113,817), and the conic differs in 58,873 (52%); the largest per-scene difference is 310 ULP at the median scene and 48,925 ULP at most. So the two projection kernels compute the conic with different rounding, and the conic is the only quantity that differs. Recomputing the kept Gaussian's alpha at the pixel center in float32 from each path's own values predicts the blended/skipped pattern of the two renders in 42 of 59 scenes (for example 0.00392157 vs 0.00392156 around 1/255 = 0.00392157); at the pixel corner it predicts it in none, which fits gsplat evaluating at the pixel center. In the other 17 scenes the recomputed alpha is within 2.7e-07 (7e-05 relative) of the threshold but on the wrong side for at least one path; that is a larger miss than float32 rounding of the final value, so the recomputation is not exact enough to settle those, and why it misses was not investigated.

**Alpha-only MR1 cases (all 6 in the 100,000-scene findings).** Five are single-Gaussian cutoff flips visible only in the alpha image (alpha 0.003922; the alpha image differs by up to 3.7e-04 while rgb stays under 1e-4). One is an early stop: 41 Gaussians, final T 3.3e-04 in one render and 1.0e-04 in the other, rgb difference 9.9e-05. So the early stop also occurs between packed and unpacked.

### Root causes (`soc-cluster-root-cause.sh`)

**Why conics differ: conditioning amplifies last-bit differences.** gsplat computes a conic as `glm::inverse` of the blurred 2x2 projected covariance, and the unpacked (`ProjectionEWA3DGSFused.cu`) and packed (`ProjectionEWA3DGSPacked.cu`) kernels run the same source for it (`persp_proj`, `add_blur`, `glm::inverse`). Inverting a 2x2 matrix multiplies a relative input rounding of about 6e-8 by its condition number kappa: a measure of how sensitive the matrix is, the ratio of its larger to smaller eigenvalue, which for the 2D footprint is the squared axis ratio. A round footprint has kappa near 1 (rounding errors stay small); a thin needle has a large kappa (the same rounding error becomes large). Kappa = 10,000 means a relative error of 6e-8 can become about 6e-4. Measured over 77,830 Gaussians in 200 scenes, the conic distance between two renders in ULP, per kappa decade (median / 90th percentile):

| kappa (condition number) | n | packed vs unpacked | scales moved by ONE float32 step | base vs scaled k = 0.01 | k = 0.25 (exact) |
| --- | --- | --- | --- | --- | --- |
| 1-10 | 76,440 | 0 / 3 | 3 / 10 | 4 / 12 | 0 / 0 |
| 10-100 | 1,377 | 1 / 7 | 4 / 14 | 5 / 17 | 0 / 0 |
| 100-1000 | 12 | 28 / 71 | 51 / 301 | 66 / 229 | 0 / 0 |
| 1000-10,000 | 1 | 0 | 115 | 450 | 0 |

So a last-bit difference of any origin (here: the smallest possible change to the scales) produces conic differences of the same size and the same growth with kappa as the packed-vs-unpacked and the world-scale differences, and none at all when the scaling is exact (k = 0.25). The relative error of the float32 conic against a float64 reference is 3.7e-07 at the median for kappa below 10 and up to about 2e-05 at 100-1000 (for both paths alike). The maxima of 6e4 to 8e5 ULP in the lowest kappa decade are not shown to be anything but an off-diagonal conic entry close to zero, whose ULP distance is huge.

**The 280-pixel Gaussian has kappa = 49,431 (axis ratio 222).** Its float32 conic is 2.5e-04 (unpacked) and 1.4e-03 (scaled, k = 100) away from the float64 reference, with 17,263 ULP between base and scaled and 299 ULP between packed and unpacked: the order of kappa x 6e-8 (3e-03). That explains why this one Gaussian changes the alpha of 280 pixels by more than the tolerance.

**The two kernels diverge in `covarW2C`, from the same source.** The unpacked and packed kernels run identical source (`quat_scale_to_covar_preci`, `covarW2C`, `persp_proj`, `add_blur`, `glm::inverse`), and the built extension (`-O3 -use_fast_math`) has the same number of floating-point instructions in both (170 `FFMA`, 136 `FMUL`, 38 `FADD`) in a different order and operand placement. To find where the bits first differ (`gsplat_mt/isolate_projection_stage.py`, 100 scenes, 26,302 Gaussians, 45.8% of conics differ between the real renders):

| Experiment | Result |
| --- | --- |
| Rebuild the two real `.cu` files unmodified, each as its own translation unit with gsplat's flags | conics equal gsplat's own in 100.0% (both kernels): the rebuild is faithful |
| Same, plus a store of the blurred 2D covariance and determinant | still 100.0% faithful; the stored values differ between the kernels in 61.7%; where they are bit-identical, the conics never differ (0 cases), so `glm::inverse` adds nothing of its own |
| Same, plus a store of the 2D covariance before the blur | still 100.0% faithful; differs in 65.3% (per entry: xx 679, xy 7,040, yx 8,647, yy 8,844 of 26,302) |
| Same, plus a store of the world covariance or the camera-space covariance | not faithful: world covariance, 87% (fused) and 48% (packed) of conics equal gsplat's; camera-space covariance, 67% (fused; packed unaffected). A stored intermediate cannot be fused away, so the store changes the compiled arithmetic. Those points were not used for comparing the kernels |
| Both real kernels given the same precomputed covariance (skipping `quat_scale_to_covar_preci`) | 45.8% of conics still differ (identical count to the run with quats and scales) |
| Both real kernels with the camera rotation set to the identity (so `covarW2C` is exact), precomputed covariance | 0 of 33,926 conics differ; with the scene's own rotation, 45.8% do |

The identity-rotation result isolates the step: when `covarW2C = R * covarW * transpose(R)` is exact, `persp_proj`, `add_blur` and `glm::inverse` give bit-identical results in the two kernels, so none of them is the origin; with a general rotation the two kernels already disagree in the camera-space covariance, and everything downstream inherits and amplifies that. `covarW2C` is two chained 3x3 matrix products, each entry a sum of three products. How a compiler pairs those products in fused multiply-adds changes the last bit of each entry, and the two kernels are compiled with different pairings (same instruction counts, different operand placement). **The pairing, found by exact emulation.** Storing the camera-space covariance changes the fused kernel's arithmetic, so it cannot be read directly. Instead, 20,000 Gaussians were placed exactly on the optical axis and the camera was rotated about z only (24 random angles, 480,000 Gaussian-by-camera samples; `isolate_projection_stage.py --pairing`). Then everything after `covarW2C` in `persp_proj` is exact multiplications that can be reproduced in software (checked on the identity rotation: 20,000 of 20,000 Gaussians), and every sum in `R * covarW * transpose(R)` has two non-zero terms. A sum `a*b + c*d` can be evaluated as unfused (both products rounded), as `fma(a, b, round(c*d))`, or as `fma(c, d, round(a*b))`. Matching the kernels' 2D covariances bit for bit against all 81 combinations per row of the 2x2 block:

| Kernel | Pattern in all 8 two-term sums (both rows, both products of `(R S) R^T`) | Samples reproduced exactly |
| --- | --- | --- |
| unpacked (`ProjectionEWA3DGSFused`) | second product rounded, first product fused into the add | 100.00% |
| packed (`ProjectionEWA3DGSPacked`) | first product rounded, second product fused into the add | 100.00% |

The best alternative combinations reproduce 85-87% of samples. The instrumented kernels still matched gsplat's own conics on this input (20,000 of 20,000 for both). So the two kernels fuse opposite products of each sum in `covarW2C`, which changes the last bit of each entry; the last-bit difference is then amplified by the conditioning of the 2D covariance in `glm::inverse`. The real sums have three terms. With a general rotation (all nine entries non-zero) the covariance was zeroed outside one index block at a time ({0,1}, {0,2}, {1,2}) so that the third product is exactly zero and the two-term test applies to that pair of positions (24 random rotations per block, 240,000 samples per block; `--three-term`). Every block reproduces both kernels at 100.00% (the best alternative 85-87%), in both rows and in both products of `(R S) R^T`:

| Position pair | unpacked (`ProjectionEWA3DGSFused`) | packed (`ProjectionEWA3DGSPacked`) |
| --- | --- | --- |
| (1st, 2nd product) | 2nd rounded, 1st fused | 1st rounded, 2nd fused |
| (1st, 3rd product) | 1st rounded, 3rd fused | 1st rounded, 3rd fused |
| (2nd, 3rd product) | 2nd rounded, 3rd fused | 2nd rounded, 3rd fused |

The pair tests fix the order of the products only for the entries that see all three position pairs (`(R S)[r][0]` sees (0,1) and (0,2); `(R S)[r][1]` sees (0,1) and (1,2); `(R S)[r][2]` sees (0,2) and (1,2); `(R S) R^T` sees all three), and they cannot tell whether the third product is fused or added separately, since it is zero there. So the full three-term sums were then searched directly (`--three-term-full`; 24 rotations, 480,000 Gaussian-by-camera samples, all four 2D covariance elements of a sample must match): for each of the five sums per row, one of the six fused chains (the order in which the three products enter a chain of multiply-adds).

| Sum | unpacked (`ProjectionEWA3DGSFused`) | packed (`ProjectionEWA3DGSPacked`) |
| --- | --- | --- |
| `(R S)[r][0]`, `(R S)[r][1]` | 2nd product rounded, 1st fused, 3rd fused | 1st rounded, 2nd fused, 3rd fused |
| `(R S)[r][2]` | 1st rounded, 2nd fused, 3rd fused | 1st rounded, 2nd fused, 3rd fused |
| `(R S R^T)[r][0]`, `[r][1]` | 2nd rounded, 1st fused, 3rd fused | 1st rounded, 2nd fused, 3rd fused |
| Samples reproduced exactly | 100.000% | 100.000% |

The best alternative assignments reproduce 87-88%, and a single pattern for every sum reproduces 79.4% (unpacked) and 100% (packed). So the packed kernel evaluates every sum in source order, and the unpacked kernel starts the chain from the second product in all sums except the third column of `R S`, where it uses source order. That is the rounding difference between the two kernels. It changes the last bit of the entries of the camera-space covariance, which the conditioning of the 2D covariance then amplifies in `glm::inverse`.

**Blend/skip decisions are reproduced exactly by the kernel's own arithmetic.** gsplat's rasterizer evaluates `sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy` and `alpha = min(0.999f, opac * __expf(-sigma))`. Compiling that expression with the same nvcc and applying it to each path's own float32 conic predicts which path blended the Gaussian in 59 of 59 MR1 scenes, with and without `-use_fast_math`, against 42 for torch float32 and 41 for a float64 evaluation. So in those scenes the decision follows from the two paths' float32 conics through the kernel's arithmetic; torch's recomputation missed 17 because of how torch evaluates that expression, not because of an unknown effect. In the two early-stop scenes the same compiled replay stops the pixel in the scaled render and blends the last Gaussian in the base render, matching the kernel (final T 7.7e-03 and 1.0e-02 against 1.0e-04).

### Every finding of the 100,000-scene run, bisected

The samples above are superseded by bisecting all of them (`diagnose_mr4.py` / `diagnose_mr1.py --scenes-from`, split into several jobs; for the 52 MR4 scenes whose difference is visible only in the alpha image, `--by alpha`).

| Group | Cutoff flip (1 Gaussian) | Depth-order flip | Early stop | Smooth conic difference | Other |
| --- | --- | --- | --- | --- | --- |
| MR4 `isolated_discontinuity` (4,643) | 4,507 | 60 | 71 | 5 | 0 |
| MR4 `tie_ambiguous` (134) | 8 | 121 | 4 | 1 | 0 |
| MR1 `isolated_discontinuity` (532) | 525 | 0 | 7 | 0 | 0 |

- **Cutoff flips**: one Gaussian with alpha at 1/255 is blended in one render and skipped in the other; the difference is at most 3.9e-03. They are 97% of the MR4 isolated cases and 99% of the MR1 ones. 33 MR4 and 5 MR1 cases show up only in the alpha image.
- **Early stops**: more than one Gaussian is needed and final T is near 1e-4. They occur in all three relations; 18 of the 71 MR4 isolated cases are alpha-only.
- **Depth-order flips**: nearly all tied scenes (121 of 134). The 60 isolated ones are one pixel with a small effect.
- **Smooth conic differences**: one Gaussian, nothing blended or skipped, final T changes by only 2e-04 to 3e-04; all are `anisotropic`.
- **Cases first counted as other**: for 5 MR4 isolated cases the bisection (which removes Gaussians while the worst pixel keeps at least half of its original difference) could not get below 2 to 13 Gaussians; removing any more of them dropped the pixel's difference below half. Their final T is not near the early-stop threshold, but their final T changes by 3.9e-03, the cutoff-flip signature. Checking each kept Gaussian's alpha at the pixel center (`diagnose_open_items.py --item 6`) shows exactly one at 1/255 in both renders and the rest far from the cutoff and identical (differences below 1e-6). So all 5 are single cutoff flips; the other Gaussians were kept because they contribute to the pixel's color difference or set the transmittance at the flipped one, and so scale it. They are counted under cutoff flips in the table.
- **The tie relabel** is right for 121 of the 134 `tie_ambiguous` scenes (90%). The other 13 (8 cutoff flips, 4 early stops, 1 smooth difference) are violations of other mechanisms that the relabel hides.

**The smooth difference is a conditioning effect, tested directly.** For the one Gaussian of the N = 27 scene (kappa = 10,625, axis ratio 103), moving every scale by a single float32 step changes its conic by 10,623 ULP, against 24,610 ULP for the world scaling at k = 0.01. The smallest possible change to the scales (1 ULP) therefore moves the conic by about kappa ULP (10,623 against kappa = 10,625). Together with the exact factor giving no difference, and the kappa table above, this closes the question for the MR4 smooth differences; for MR1 the input difference is the one inside `covarW2C`.

### Limits

- The multiply-add pairing was measured on one build of the extension (the cached gsplat 1.5.3 build, `-O3 -use_fast_math`, one nvcc) and one input configuration (Gaussians on the optical axis at camera depth 4, so that `persp_proj` is exact multiplications). It was run on both H100 variants of the cluster (`h100-96` and `h100-47`) and gave identical results (for example the single-pattern search reproduces 79.355% of the unpacked kernel's samples on both). Other compiler versions were not tested and may select different multiply-add pairings. But comparing this would be beyond the scope of this project and beyond my capability.
