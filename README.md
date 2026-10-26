# gsplat Metamorphic Testing

Metamorphic testing framework for `gsplat`'s forward RGB rasterizer: a
synthetic Gaussian scene generator (default plus edge-case modes) and five
metamorphic relations (packed/unpacked, batched/individual cameras, covariance
vs scale+quaternion, uniform world scale, Gaussian permutation), each judged
pass/violation/crash against a tolerance. Requires an NVIDIA GPU as gsplat's rasterizer is a
CUDA-only extension.

## Setup

```bash
conda create -n gsplat python=3.12
conda activate gsplat
pip install -r requirements.txt # make sure you are within a GPU enabled environment (alloc -G a100-40)
```

## Run

```bash
python gsplat_mt/run_preliminary.py --n-scenes 20 --out results/refined_results.csv
```

Useful flags (`--help` for the full list):

- `--n-scenes`: scenes per mode
- `--modes`: subset of `default opacity_extreme scale_tiny scale_huge anisotropic wide_spread far_clipped single`
- `--world-scales`: MR4 scale factors k (default `0.01 0.25 4 100`)
- `--n-gaussians-min/max`, `--width/--height`, `--n-cameras`
- `--mr1-backgrounds`: also run MR1 with a background color set, to probe gsplat issue [#764](https://github.com/nerfstudio-project/gsplat/issues/764)
- `--skip-repro`: skip the standalone issue #764 / [#630](https://github.com/nerfstudio-project/gsplat/issues/630) reproduction checks

Outputs (next to `--out`): `*_results.csv` (per-scene MR results), `*_repro.json`
(issue reproduction outcomes), `*_summary.json` (aggregate D_max/D_MAE stats and per-mode verdict counts).

Tolerances (max abs pixel difference) are set in `gsplat_mt/metrics.py`: 1e-4 for MR1 and MR4, exact equality for MR2, MR3 and MR5. MR4/MR5 values are initial. MR5 differences, and MR4 differences, in scenes with near-equal depths are reported as `tie_ambiguous`, not `violation`. For MR1 and MR4, a violation confined to at most 0.05% of pixels (2 pixels at 64x64) is reported as `isolated_discontinuity`: a single pixel crossing a rasterizer cutoff such as alpha < 1/255. The raw D_max and the per-image pixel count (`*_n_px_over`) are still recorded.

Each scene also records visibility: `coverage_cam0`/`coverage_mean` (fraction of pixels with alpha > 0), `opaque_frac_cam0`/`opaque_frac_mean` (alpha > 0.99), `n_empty_cams`, and `n_visible_cam0`/`n_visible_any` (Gaussians surviving gsplat's culling). A `pass` on a render that draws nothing (camera 0 empty; for MR2, every camera empty) is reported as `vacuous_pass`.

Each of MR3, MR4 and MR5 also runs a fault-injected variant (`mr3_xyzwbug`, `mr4_bug`, `mr5_bug`), where `violation` means the injected bug was detected.

## Large-scale run

```bash
python gsplat_mt/run_large_scale.py --n-scenes 2000 --out results/large_scale.csv
python gsplat_mt/run_large_scale.py --only-scene 137 --out results/one.csv   # replay one scene
```

Each scene samples its own configuration (Gaussian count 10-5000, scene radius, scale and opacity ranges, camera distance, field of view, 1-8 cameras, resolution) and a stress mode, then runs MR1-MR5 as above. Config values are recorded as `cfg_*` columns; a mode's own overrides win over the sampled values they touch. Outputs next to `--out`: `*.csv`, `*.jsonl` (written per scene, so a timeout keeps partial results), `*_findings.csv` (scenes with any crash/violation/isolated_discontinuity/tie_ambiguous/invalid_output), `*_summary.json` (verdict counts per mode, fault-injection detection, Spearman correlations of scene properties with MR1/MR4 D_max). The run stops early on a CUDA error other than out-of-memory, since later scenes would be invalid.

## SLURM (SoC cluster)

```bash
sbatch soc-cluster.sh        # all scene modes, default settings
sbatch soc-cluster-large.sh  # large-scale randomized study
```

Each requests a single GPU, activates the `gsplat` conda env, and installs dependencies.

## Layout

- `gsplat_mt/scene.py` -- scene/camera generator and stress modes
- `gsplat_mt/metrics.py` -- D_max/D_MAE oracle, tolerances, verdicts
- `gsplat_mt/mr1_packed.py`, `mr2_batch.py`, `mr3_covariance.py`, `mr4_world_scale.py`, `mr5_permutation.py` -- the five MRs
- `gsplat_mt/repro_issues.py` -- reproductions of gsplat issues #764 and #630
- `gsplat_mt/run_preliminary.py` -- CLI runner
- `gsplat_mt/run_large_scale.py` -- randomized-configuration runner (reuses `run_preliminary.run_one_scene`)
- `gsplat_mt/diagnose_mr4.py` -- per-pixel MR4 diagnosis for one scene
