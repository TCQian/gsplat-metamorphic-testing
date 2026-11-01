#!/bin/bash
#SBATCH --job-name=gsplat-stage
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate gsplat

pip install -r requirements.txt

# MAX_JOBS=1: two large translation units are compiled; more parallel nvcc runs do not fit in 8 GB
export MAX_JOBS=1
python gsplat_mt/isolate_projection_stage.py --n-scenes 100 --identity-rot --covars --pairing --three-term --three-term-full --stages 0 1 2 3 4
