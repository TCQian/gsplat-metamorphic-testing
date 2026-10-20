#!/bin/bash
#SBATCH --job-name=gsplat-testing
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate gsplat

pip install -r requirements.txt
python gsplat_mt/run_preliminary.py --n-scenes 20 --out results/week8_results.csv
