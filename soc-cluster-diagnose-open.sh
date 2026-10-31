#!/bin/bash
#SBATCH --job-name=gsplat-open
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate gsplat

pip install -r requirements.txt

echo "=== 1-5. open cases: smooth difference, 17-Gaussian violation, 280-pixel cutoff flip, early stop, alpha-only MR1"
python gsplat_mt/diagnose_open_items.py --item all --csv results/xl_scale_findings.csv

echo "=== 6. MR1 cutoff flips of the 2000-scene study (all): packed vs unpacked projection outputs, bit for bit"
python gsplat_mt/diagnose_mr1_inputs.py --scenes-from results/large_scale_w11.csv --where mr1_verdict=isolated_discontinuity

echo "=== 7. MR1 cutoff flips of the 100,000-scene run (every 10th)"
python gsplat_mt/diagnose_mr1_inputs.py --scenes-from results/xl_scale_findings.csv --where mr1_verdict=isolated_discontinuity --every 10
