#!/bin/bash
#SBATCH --job-name=gsplat-diagnose
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate gsplat

pip install -r requirements.txt

python gsplat_mt/diagnose_mr4.py --scene-idx 68 --mode scale_huge --bisect
for i in 112 444 1584 1745; do
    python gsplat_mt/diagnose_mr4.py --scene-idx $i --large --bisect
done
