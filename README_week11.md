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

## Checked so far

- `verdict()` logic checked locally with a stub (no GPU): a 3.5e-3 one-pixel difference is `isolated_discontinuity`, a 5.5e-2 one is `violation`, 28 pixels at 2e-3 is `violation`, any nonzero difference at tolerance 0 is `violation`; `is_power_of_two` is true for 1/128, 1/4, 4, 128 and false for 0.01 and 100.
- Everything else (bisection on the new scripts, `meta` indexing for packed renders, the probe) has not been run.

## Read from gsplat 1.5.3 source (`IntersectTile.cu`, `RasterizeToPixels3DGSFwd.cu`)

- Each Gaussian-tile pair gets a 64-bit key: image id | tile id | the depth's 32 float bits, reinterpreted as an integer (no quantization), and the pairs are sorted by a CUB radix sort, which is stable. So two Gaussians with exactly equal depth in a tile keep the order in which their pairs were generated, and depths one float32 step apart are always ordered by depth. That generation order is not read from the source here; `probe_ties.py` tests whether it follows the Gaussian index.
- The early stop is exclusive: when `next_T = T * (1 - alpha)` is at most 1e-4, the pixel stops and that Gaussian is not blended. So a pixel whose alpha rounds slightly differently can skip a Gaussian with a large alpha (a large change in T) while changing the color by at most about `color x alpha x T` with T below about 1e-4 / (1 - alpha).

## Results

Not yet run on the cluster.
