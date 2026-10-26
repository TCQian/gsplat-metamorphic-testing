#!/bin/bash
#SBATCH --job-name=gsplat-w11
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate gsplat

pip install -r requirements.txt

CSV=results/large_scale_w11.csv

echo "=== 1. rerun of the 2000-scene study with the exact-zero MR4 check and the isolated-size bound"
python gsplat_mt/run_large_scale.py --n-scenes 2000 --out $CSV

echo "=== 2. how gsplat orders equal and one-step-apart depths"
python gsplat_mt/probe_ties.py

echo "=== 3. MR4: bisect every remaining isolated_discontinuity scene"
python gsplat_mt/diagnose_mr4.py --scenes-from $CSV --where mr4_verdict=isolated_discontinuity

echo "=== 4. MR4: bisect every tie_ambiguous scene"
python gsplat_mt/diagnose_mr4.py --scenes-from $CSV --where mr4_verdict=tie_ambiguous

echo "=== 5. MR4: the three former violators, pixel with the largest transmittance change"
for i in 112 444 1584; do
    python gsplat_mt/diagnose_mr4.py --scene-idx $i --large --bisect --by T
done

echo "=== 6. MR1: bisect every isolated_discontinuity scene (packed vs unpacked)"
python gsplat_mt/diagnose_mr1.py --scenes-from $CSV --where mr1_verdict=isolated_discontinuity
