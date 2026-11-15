# gsplat Metamorphic Testing

Metamorphic testing of the forward RGB rasterizer of [`gsplat`](https://github.com/nerfstudio-project/gsplat) (pinned `gsplat==1.5.3`). Random scenes of 3D Gaussians are rendered two ways that must give the same image, and the two images are compared. No reference images are needed. Five metamorphic relations are tested:

| Relation | The two renders | Expected |
| --- | --- | --- |
| MR1 | `packed=True` vs `packed=False` | equal within 1e-4 |
| MR2 | all cameras in one call vs one call per camera | identical |
| MR3 | precomputed covariance vs scale + quaternion | identical |
| MR4 | means, scales, camera translation and near/far all scaled by k | equal within 1e-4; identical for power-of-two k |
| MR5 | Gaussians shuffled | identical, unless two depths are tied |

Every relation renders every camera of a scene in one batched call. MR3, MR4 and MR5 also run a planted fault (a deliberate mistake in the test itself: an xyzw quaternion, an unscaled camera translation, colors not shuffled with the Gaussians), which must be detected.

## Results

100,000 random scenes (nine scene types, 1-8 cameras, up to 5,000 Gaussians, up to 256x256), 102 minutes on one H100:

- No crashes and no wrong images. MR2, MR3 and MR4 with power-of-two k are bit-exact in all 99,628 non-empty scenes.
- MR5 changes the image only in scenes with near-equal depths (2,583 of 29,955 such scenes, 0 of 69,673 others).
- MR1 and MR4 show small differences in about 2% and 14% of scenes, almost always a few pixels of one camera's image. Every one examined is float32 rounding (largest for needle-shaped Gaussians) turned visible by one of gsplat's hard thresholds: the 1/255 alpha cutoff, the 1e-4 transmittance early stop, or depth order.
- The planted faults are detected in every non-empty scene, except the shuffle fault where the random shuffle leaves every visible Gaussian in place; a shuffle with no fixed points detects those too.

The full analysis, with its caveats, is in [`docs/concluding_report.md`](docs/concluding_report.md).

## Setup

gsplat's rasterizer is a CUDA-only extension, so an NVIDIA GPU is required (no CPU or MPS backend).

```bash
conda create -n gsplat python=3.12
conda activate gsplat
pip install -r requirements.txt
```

The first gsplat call JIT-compiles the CUDA extension (about 12 minutes, once per environment).

## Run

On a SLURM cluster (the scripts request an H100 on the NUS SoC cluster; change `--gres` for another cluster), from the repository root:

```bash
sbatch soc-cluster.sh                       # preliminary + large-scale study
sbatch soc-cluster.sh preliminary           # 20 small scenes per scene type, a few minutes
sbatch soc-cluster.sh large                 # 100,000 randomized scenes, about 100 minutes
sbatch soc-cluster-diagnose.sh violations   # then diagnose the findings; steps are listed in the script
```

On any CUDA machine, from the repository root:

```bash
python runners/run_preliminary.py --n-scenes 2 --modes default single --skip-repro --out results/smoke.csv   # quick check
python runners/run_preliminary.py --n-scenes 20 --out results/preliminary.csv
python runners/run_large_scale.py --n-scenes 100000 --out results/allcam_100000.csv
python runners/run_large_scale.py --only-scene 4896 --out results/one.csv   # replay one scene
```

`--help` lists every option. The large-scale defaults (seed 2000, all nine scene types) reproduce the run above. Scene types: `default opacity_extreme scale_tiny scale_huge anisotropic wide_spread far_clipped single dense_cover`. `--world-scales` sets the MR4 factors (default `0.01 0.0078125 0.25 4 100 128`).

## Outputs

`results/` (not tracked) receives, next to `--out`:

- `<out>.csv`: one row per scene: scene type and properties, the sampled configuration (`cfg_*`, large-scale only), and per relation D_max, D_MAE, D_SSIM, pixels over tolerance and a verdict.
- `<out>_findings.csv` (large-scale): scenes with any `crash`, `violation`, `isolated_discontinuity`, `tie_ambiguous` or `invalid_output`.
- `<out>_summary.json`: verdict counts per scene type, planted-fault detection, timing and Spearman correlations of scene properties with MR1/MR4 D_max.
- `<out>.jsonl` (large-scale): written scene by scene, so a timeout keeps partial results.
- `<out>_repro.json` (preliminary): standalone reproductions of gsplat issues [#764](https://github.com/nerfstudio-project/gsplat/issues/764) (`packed=True` with `backgrounds`) and [#630](https://github.com/nerfstudio-project/gsplat/issues/630) (xyzw quaternion mislabel, a caller-side bug).

Verdicts (`gsplat_mt/metrics.py`):

| Verdict | Meaning |
| --- | --- |
| `pass` | within tolerance (1e-4 for MR1 and MR4, exactly 0 otherwise) |
| `isolated_discontinuity` | MR1/MR4 only: at most 0.05% of the pixels of every camera image over tolerance, and D_max at most 4e-3 (the most one Gaussian crossing the 1/255 alpha cutoff can change a pixel) |
| `tie_ambiguous` | a `violation` of MR4 or MR5 in a scene where some camera sees two Gaussians with depths closer than 1e-6 |
| `violation` | anything else over tolerance; for a planted fault it means the fault was detected |
| `vacuous_pass` | a pass where every camera's image is empty |
| `crash` / `invalid_output` | gsplat raised an error / both renders agree but contain NaN or Inf |

`mr4_exact_verdict` checks the power-of-two factors with tolerance 0 and is never relabeled. Per-scene coverage columns (`coverage_mean`, `coverage_min`, `n_covered_px`, `opaque_frac_mean`, `n_visible_any`) show how many pixels a pass rests on.

## Layout

```
gsplat_mt/                library
  scene.py                random Gaussian scenes, cameras and the nine scene types
  metrics.py              image comparison (D_max, D_MAE, SSIM), tolerances, verdicts
  relations/              MR1-MR5 (mr1_packed.py ... mr5_permutation.py), with planted faults
  harness.py              one scene through every relation -> one result row
  sampling.py             per-scene random configuration of the large-scale run; scene replay
  repro_issues.py         reproductions of gsplat issues #764 and #630
runners/
  run_preliminary.py      fixed settings, a number of scenes per scene type
  run_large_scale.py      every scene draws its own configuration
diagnosis/                replay scenes from a results CSV and explain findings
  diagnose_mr4.py         bisect (delta debugging) the Gaussians behind one MR4 pixel
  diagnose_mr1.py         the same for MR1 (packed vs unpacked)
  diagnose_allcam.py      violations at their worst camera; missed shuffle faults; dense_cover coverage
  study_undetected_faults.py   why a planted fault was missed
  probe_ties.py           how gsplat orders equal depths
  diagnose_conic_precision.py, diagnose_mr1_inputs.py, diagnose_open_items.py,
  replicate_kernel_arithmetic.py, isolate_projection_stage.py
                          root-cause tools for the earlier camera-0 runs (seed 1000)
docs/
  concluding_report.md    results, root causes, usefulness
  progress/week07-13.md   weekly progress notes
soc-cluster.sh            SLURM: preliminary and large-scale study
soc-cluster-diagnose.sh   SLURM: diagnosis steps
```

The diagnosis tools replay a scene from the run's seed and scene types (`--seed`, `--modes`), so they need the same settings as the run that produced the CSV. They default to the all-camera run, except the five root-cause tools of the camera-0 runs, which default to seed 1000 and the original eight scene types.
