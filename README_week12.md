# Week 12: perceptual-diagnostic response to reviewer feedback

Feedback on §3.3 (Metamorphic Oracle) proposed a human-vision model to address floating-point noise, citing the reviewer's own ISSTA'24 paper ("Inconsistencies in TeX-Produced Documents", Tan & Rigger, doi:10.1145/3650212.3680370).

## "Human vision model"

Compares images by perceived difference, not raw pixel arithmetic. From cheap to literal:

- **SSIM/MS-SSIM** -- implemented below; a simplified HVS proxy.
- **LPIPS** -- learned from human judgments; standard in NeRF/3DGS evaluation.
- **HDR-VDP-2 / FLIP** -- simulate optics/contrast sensitivity to predict visibility.

## That paper's actual finding

They tried SSIM/MSE first and dropped them: both flagged nearly every TeX-engine page pair as different due to reader-invisible noise (hyphenation, antialiasing) -- the same false-positive risk floating-point noise poses here. They ended up using SIFT feature matching plus manual categorization instead. Takeaway: validate a perceptual signal against known-noise vs. known-bug cases before trusting it.

## Changes (`gsplat_mt/`)

- `metrics.py`: added `ssim()` (Wang et al. 2004, 11x11 Gaussian window, pure `torch`, no new dependency, computed in float64 so that `1 - d_ssim` of about 1e-7, the size of a single cutoff-flip pixel, is not lost to float32 rounding near 1.0) and a `d_ssim` field on `ImageDiff`, so every MR comparison now records `*_d_ssim`. Diagnostic only -- `verdict()` unchanged.
- `run_preliminary.py` / `run_large_scale.py`: MR2's manual row and the numeric-key summaries now include `d_ssim`.
- `repro_issues.py`: issue #630 repro also prints D_ssim.

`main.tex`/`ref.bib` not touched -- the report/bib additions are the user's.

## Validation

Re-implemented the same formula standalone in numpy on synthetic 64x64 images, to check the formula before trusting the torch path:

| Case | D_SSIM |
| --- | --- |
| Identical | 1.000000 |
| Tiny float noise (MR1-scale) | 1.000000 |
| One pixel, +3.9e-3 (isolated-cutoff bound) | 1.000000 |
| 32x32 region replaced | 0.760972 |
| One pixel, D_max=0.264 (xyzw-bug scale) | 0.999906 |
| 32x32 region, same D_max=0.264 | 0.967888 |

Same D_max (0.264), very different D_SSIM -- confirms the formula separates "one pixel" from "a whole region" as intended.

## Results on the cluster (2000 scenes + the five untied MR4 violations)

`1 - d_ssim` (float64), by what the verdict says:

| Group | n | `1 - d_ssim` range (median) |
| --- | --- | --- |
| MR1 pass | 1969 | 0 to 4.9e-11 (6.7e-16) |
| MR1 `isolated_discontinuity` (cutoff flips) | 7 | 4.6e-11 to 3.0e-08 (3.7e-09) |
| MR4 pass (worst k) | 1888 | 0 to 2.6e-09 (3.6e-13) |
| MR4 `isolated_discontinuity` | 84 | 7.5e-11 to 1.5e-07 (1.1e-08) |
| MR4 depth-order violators (`tie_ambiguous`) | 4 | 8.2e-08 to 1.2e-06 |
| MR5 pass / `tie_ambiguous` | 1960 / 16 | 0 / 0 to 2.3e-07 |
| MR2, MR3 correct, MR4 exact | 2000 each | exactly 0 |
| Injected MR3 `xyzw_bug` | 1976 | 3.3e-13 to 0.50 (6.0e-03) |
| Injected MR4 `no_translation` | 1977 | 3.0e-07 to 1.0 (0.34) |
| Injected MR5 `misaligned` | 1712 | 9.6e-08 to 0.72 (5.4e-02) |
| The five MR4 violations in untied scenes | 5 | 1.8e-09 to 1.2e-06 |

- Float64 matters: the benign differences are 1e-11 to 1e-7 from 1, below what a float32 mean near 1.0 resolves.
- On medians the groups order as expected (noise < cutoff flips < depth-order flips < injected bugs), but the ranges overlap:
  - the largest MR4 pass (2.6e-09) is above the smallest isolated case (7.5e-11);
  - the largest isolated case (1.5e-07) is above the smallest depth-order violator (8.2e-08);
  - the weakest injected MR4 and MR5 bugs (3.0e-07, 9.6e-08) sit inside the range of benign findings;
  - 103 of 1976 `xyzw_bug` scenes have `1 - d_ssim` of at most 1e-8, in the noise and cutoff-flip range.
- The five untied violators are not separated from benign findings: all five lie within 1.8e-09 to 1.2e-06. The two largest (7e-03 and 9e-03 D_max) have 4.9e-07 and 1.2e-06; the smallest (D_max 2.3e-04) has 1.8e-09, inside the MR4 pass range.
- D_max with the existing tolerances already flags every injected bug in a non-empty scene (MR3 is exact, so any nonzero difference counts; the smallest injected D_max is 2.9e-03 for MR4 and 1.6e-03 for MR5), so SSIM adds no detection. It does carry extent information (a region versus a pixel) that D_max does not.

So the condition set above ("clean separation") is not met: `verdict()` stays unchanged and SSIM stays a diagnostic column.

## Next

- Nothing further planned for SSIM. LPIPS and FLIP were not tried.
