#!/bin/bash
#SBATCH --job-name=gsplat-diag-xl
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate gsplat

pip install -r requirements.txt

CSV=results/xl_scale_findings.csv

echo "=== 1. MR4: every violation (none has a depth tie)"
python gsplat_mt/diagnose_mr4.py --scenes-from $CSV --where mr4_verdict=violation

echo "=== 2. MR4: every other tie_ambiguous scene"
python gsplat_mt/diagnose_mr4.py --scenes-from $CSV --where mr4_verdict=tie_ambiguous --every 2

echo "=== 3. MR4: every 40th isolated_discontinuity scene"
python gsplat_mt/diagnose_mr4.py --scenes-from $CSV --where mr4_verdict=isolated_discontinuity --every 40

echo "=== 4. MR1: every 10th isolated_discontinuity scene"
python gsplat_mt/diagnose_mr1.py --scenes-from $CSV --where mr1_verdict=isolated_discontinuity --every 10
