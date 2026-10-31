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

## Next

- Run on the cluster to check real data: does `d_ssim` stay ~1.0 on the known `isolated_discontinuity`/`tie_ambiguous` cases while dropping on the five untied MR4 violators and the xyzw-bug repro? Not started -- needs approval.
- Leave `verdict()` alone until that confirms clean separation.
