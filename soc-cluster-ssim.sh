#!/bin/bash
#SBATCH --job-name=gsplat-ssim
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate gsplat

pip install -r requirements.txt

echo "=== 1. 2000-scene study with *_d_ssim recorded (passes, cutoff flips, tied scenes, fault injections)"
python gsplat_mt/run_large_scale.py --n-scenes 2000 --out results/ssim_2000.csv

echo "=== 2. the five MR4 violations in untied scenes (scene indices from the 100,000-scene run)"
for i in 28499 53078 65342 65923 85315; do
    python gsplat_mt/run_large_scale.py --only-scene $i --out results/ssim_one_$i.csv
done
