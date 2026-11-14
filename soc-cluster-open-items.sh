#!/bin/bash
#SBATCH --job-name=gsplat-open
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate gsplat

pip install -r requirements.txt

# Experiments on the open items of the 100,000-scene all-camera run (job 1102754; seed 2000, nine modes).

# 1. Does a shuffle with no fixed points detect the undetected MR5 faults?
python gsplat_mt/experiments_open_items.py derangement --undetected results/undetected_allcam_100000.csv --seed 2000

# 2. The affected Gaussians of the 'contribute nothing' scenes: hidden by others, or never reach a pixel?
python gsplat_mt/experiments_open_items.py nothing --undetected results/undetected_allcam_100000.csv --seed 2000

# 3. Why does dense_cover cover 44% and not the predicted 85-98%?
python gsplat_mt/experiments_open_items.py coverage --csv results/allcam_100000.csv --seed 2000 --n 60

# 4. Bisect the 1 MR1 and 8 MR4 violations at the camera that shows them.
python gsplat_mt/experiments_open_items.py violations --csv results/allcam_100000.csv --seed 2000
