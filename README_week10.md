# Week 10: analysis of numerical discrepancies

What kinds of differences the metamorphic relations produced, how large and how widespread they are, what they trace back to, and which scene characteristics go with them.

## Terms

- **D_max**: the largest absolute difference between the two renders at any single pixel and color channel (pixel values are in [0, 1]). **D_MAE** is the mean over all pixels, so large D_max with tiny D_MAE means a few pixels differ a lot.
- **Blending**: gsplat sorts the Gaussians covering a pixel by depth and blends them front to back. Each adds `color x alpha x T`, where T (transmittance) is the light still passing through; T is multiplied by `(1 - alpha)` after each Gaussian.
- **Alpha cutoff**: a Gaussian whose alpha at a pixel is below 1/255 (about 0.0039) is skipped at that pixel (`ALPHA_THRESHOLD` in gsplat's `cuda/include/Common.h`).
- **Power-of-two factor**: scaling by 2^n (1/128, 1/4, 4, 64, 128, ...) is exact in binary floating point, so no rounding is introduced. Factors such as 0.01 and 100 are not exact.
- **Isolated discontinuity**: the verdict for MR1/MR4 when a tolerance failure touches at most 0.05% of the pixels.

## Changes

### Diagnosis tool (`gsplat_mt/diagnose_mr4.py`)

Replays one scene and compares the original render with the k-scaled render pixel by pixel.

- `--scene-idx N --mode MODE` replays a `run_preliminary.py` scene; `--large` replays a `run_large_scale.py` scene with its sampled configuration.
- Sweeps exact (power-of-two) and inexact k. For pixels over tolerance it reports how many 16x16 tiles they fall in, the relative change in final transmittance, the alpha of the single Gaussian that would explain a change (`alpha = 1 - T_small / T_large`), and the smallest relative depth gaps between depth-adjacent Gaussians that could cover the pixel.
- `--bisect [--bisect-k K]`: for the worst pixel, removes Gaussians chunk by chunk (delta debugging), keeping each removal that still leaves at least half of the original difference at that pixel. It ends with a smallest set of Gaussians that causes the difference and prints their depths and alphas in both renders, from gsplat's own reported values (`meta["depths"]`, `meta["conics"]`, `meta["opacities"]`). The search logic was checked on cases with a known answer (it found the exact needed Gaussians in all four).

## What was tested

- Every discrepancy from the 2000-scene randomized run was grouped by relation, size (D_max), and number of pixels over tolerance.
- Scene characteristics were compared against discrepancies: number of Gaussians, scale, opacity, depth, overlap, coverage, resolution, camera count, and stress mode (rank correlations, rates by size bucket and by mode).
- Selected scenes were replayed at pixel level with exact and inexact scale factors: one small `scale_huge` scene and three large violating scenes (two `default`, one `anisotropic`), plus the largest-D_max scene (`scale_tiny`).
- The MR4 results were cross-checked against MR5's depth-tie flag.

## Findings

### What the discrepancies are

| Relation | Over tolerance | What it looks like |
| --- | --- | --- |
| MR2, MR3 (correct) | none | D_max exactly 0 in every scene |
| MR1 packed vs unpacked | 7 of 2000 | exactly one pixel each, max D_max 1.9e-03, all with 1300+ Gaussians |
| MR5 permutation | 16 of 2000 | every one has near-equal depths; no scene without a depth tie differs |
| MR4 world scale | 88 of 2000 | 85 touch at most 0.024% of pixels; 3 touch 0.21-0.28% |

- The factor with the largest MR4 difference was always 0.01 or 100, the two that are not powers of two.
- In all four replayed scenes, every power-of-two factor gave D_max exactly 0 over all pixels and repeated renders were identical. gsplat's scale handling is exact there, so the differences come from rounding at the inexact factors.

### A single alpha-cutoff flip is limited to about 1/255

If one Gaussian is blended in one render and skipped in the other, the pixel changes by at most `color x alpha x T`, and alpha is about 1/255, so at most about 3.9e-03.

- Direct measurement on a small `scale_huge` scene: at the one differing pixel, final transmittance differed by 0.393% and the implied alpha was 0.00393 (1/255 = 0.00392).
- Of the eight largest MR4 D_max values in the run, six single-pixel cases lie between 3.2e-03 and 3.6e-03, just under that bound. One of them is an N = 1 scene: a lone Gaussian cannot be reordered with anything, so its single-pixel difference at the edge of its footprint is the cutoff effect in its simplest form.
- Two of the eight are above the bound: a `default` scene at 4.9e-03 (185 pixels) and a `scale_tiny` scene at 5.5e-02 (5 pixels, 14x the bound).

### The three MR4 violations differ from the single-pixel cases

All three are at k = 0.01, have 1400+ Gaussians, and touch 28 to 185 pixels (0.21-0.28% of the image), D_max 1.8e-03 to 4.9e-03, D_MAE around 1e-6, and the pixels fall in 1-4 tiles.

- Final transmittance at the affected pixels is unchanged in two of the three scenes (max relative change 2.7e-05 and 0). So the same Gaussians contribute in both renders. In the third, most pixels are unchanged but a few change a lot (up to 77%); that pattern would fit the early-stop cutoff (blending stops once T drops below 1e-4), but this was not confirmed.
- Unchanged transmittance with a changed color is what a swap of two near-equal-depth Gaussians would produce. Evidence for that is mixed:
  - One scene has a pair at a relative depth gap of 6e-08 (about one float step) among only 30 candidate Gaussians.
  - One has a pair at exactly equal depth among 496 candidates.
  - One has a smallest gap of 9.9e-08 among about 1000 candidates; with that many candidates a gap that small is about what random depths give, so it is not evidence by itself.
  - MR5 reports exactly 0 difference on all three scenes, although all three are flagged as having depth ties. A swap would often, not always, change under a random permutation.
- The mechanism of these three is therefore not established (see Pending).

### Scene characteristics associated with larger differences

- **Number of Gaussians**: the share of scenes with MR4 D_max above 1e-4 is 0.6% for N < 100, 2.6% for N 100-999 and 15.6% for N >= 1000. The three violators have 1410 to 4402 Gaussians; the isolated cases include small scenes too (an N = 1 and an N = 13 scene are among the eight largest D_max values).
- **Stress mode** (share of that mode's scenes with MR4 D_max above 1e-4; each mode has about 230-270 scenes): `scale_huge` 11.6%, `default` 5.5%, `wide_spread` 5.8%, `anisotropic` 5.6%, `opacity_extreme` 3.5%, `scale_tiny` 2.0%, `far_clipped` 1.5%, `single` 0.4%.
- **Depth ties**: 371 of 2000 scenes have near-equal depths (303 with N >= 1000). 18.6% of all scenes are tied, but 65% of the MR4 isolated cases (55 of 85) and all 3 violators are. N is a confound, since both ties and tolerance failures become likelier with more Gaussians.
- **MR5**: among tied scenes, 12 of 303 (N >= 1000) and 4 of 68 (N < 1000) show a difference, so the share is not higher at large N.
- **Rank correlations with D_max (Spearman)**, over all 2000 scenes:

| Scene property | MR1 D_max | MR4 D_max |
| --- | --- | --- |
| overlap fraction | 0.79 | -0.36 |
| opaque fraction (camera 0) | 0.71 | 0.01 |
| coverage (camera 0) | 0.69 | 0.06 |
| number of Gaussians | 0.53 | 0.45 |
| mean scale | 0.41 | -0.41 |
| pixels (resolution) | 0.10 | 0.25 |
| mean depth | -0.14 | 0.04 |
| mean opacity | 0.06 | 0.15 |
| cameras | -0.01 | 0.01 |

  These are weak evidence. Overlap, coverage and N are strongly related to each other; most D_max values are 0 or float noise; and I have not explained the negative MR4 signs for overlap and scale.
- **Image content**: most renders are mostly background (opaque fraction at most 0.05 outside `scale_huge`), and with the original `scale_huge` range images were fully opaque (see Week 8). A pass on a sparse image rests on few pixels.

### The pixel-fraction rule is not enough

The `isolated_discontinuity` label, based only on pixel count, accepted a scene with D_max 5.5e-02, far above what one alpha-cutoff flip can produce. A label that also bounds the size of the difference (about 1/255) would not have.

## Proposed changes (not implemented)

- Add power-of-two factors to the MR4 sweep with an exact-zero tolerance. All four replayed scenes were exactly 0 there, so any difference would indicate a genuine scale-handling error, with no pixel-fraction judgment needed. Inexact factors would be informational.
- Require the `isolated_discontinuity` label to also satisfy D_max of at most about 4e-3, a bound derived from the cutoff mechanism and not tuned to this data.

## Pending

- Bisection results (smallest set of Gaussians that reproduces the worst pixel, and their depths and alphas in both renders) for the three violators, the `scale_huge` scene, and the 5.5e-02 outlier. This is what will say whether the violators are depth-order swaps, cutoff flips, or something else.
- The seven MR1 single-pixel cases and the 5.5e-02 outlier have not been diagnosed.
