#!/bin/bash
#SBATCH --job-name=gsplat-xl
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate gsplat

pip install -r requirements.txt
python gsplat_mt/run_large_scale.py --n-scenes 100000 --out results/xl_scale.csv
