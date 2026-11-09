#!/bin/bash
#SBATCH --job-name=gsplat-allcam
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate gsplat

pip install -r requirements.txt

MODES="default opacity_extreme scale_tiny scale_huge anisotropic wide_spread far_clipped single dense_cover"

# 1. Does dense_cover cover the frame? (compare coverage_mean with the other modes in the CSV)
python gsplat_mt/run_large_scale.py --n-scenes 500 --modes dense_cover --seed 2000 --out results/dense_smoke.csv

# 2. Every relation renders every camera; all nine scene types; fresh seed.
python gsplat_mt/run_large_scale.py --n-scenes 20000 --modes $MODES --seed 2000 --out results/allcam_20000.csv

# 3. Planted faults still undetected in that run (single-Gaussian scenes are trivially undetectable for MR5).
python gsplat_mt/study_undetected_faults.py --csv results/allcam_20000.csv --seed 2000 --modes $MODES \
    --skip-modes single --out results/undetected_allcam.csv

# 4. The 77 shuffle faults missed in the 100,000-scene run (camera 0 only): caught by other cameras, or what cause?
python gsplat_mt/study_undetected_faults.py --csv results/xl_scale.csv --faults mr5_bug \
    --skip-modes single --out results/undetected_xl.csv
