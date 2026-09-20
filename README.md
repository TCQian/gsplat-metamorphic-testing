# gsplat Metamorphic Testing

Metamorphic testing framework for `gsplat`'s forward RGB rasterizer: a
synthetic Gaussian scene generator plus three metamorphic relations
(packed/unpacked, batched/individual cameras, covariance vs
scale+quaternion). Requires an NVIDIA GPU as gsplat's rasterizer is a
CUDA-only extension.

## Setup

```bash
conda create -n gsplat python=3.12
conda activate gsplat
pip install -r requirements.txt # make sure you are within a GPU enabled environment (alloc -G a100-40)
```

## Run

```bash
python gsplat_mt/run_preliminary.py --n-scenes 20 --out results/preliminary_results.csv
```

Useful flags (`--help` for the full list):

- `--n-gaussians-min/max`, `--width/--height`, `--n-cameras`
- `--mr1-backgrounds`: also run MR1 with a background color set, to probe gsplat issue [#764](https://github.com/nerfstudio-project/gsplat/issues/764)
- `--skip-repro`: skip the standalone issue #764 / [#630](https://github.com/nerfstudio-project/gsplat/issues/630) reproduction checks

Outputs (next to `--out`): `*_results.csv` (per-scene MR results), `*_repro.json`
(issue reproduction outcomes), `*_summary.json` (aggregate D_max/D_MAE stats).

## SLURM (SoC cluster)

```bash
sbatch soc-cluster.sh
```

Requests a single A100-40GB GPU, activates the `gsplat` conda env, installs
dependencies, and runs the preliminary study with default settings.

## Layout

- `gsplat_mt/scene.py` -- scene/camera generator
- `gsplat_mt/metrics.py` -- D_max/D_MAE oracle
- `gsplat_mt/mr1_packed.py`, `mr2_batch.py`, `mr3_covariance.py` -- the three MRs
- `gsplat_mt/repro_issues.py` -- reproductions of gsplat issues #764 and #630
- `gsplat_mt/run_preliminary.py` -- CLI runner
