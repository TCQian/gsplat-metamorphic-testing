#!/bin/bash
#SBATCH --job-name=gsplat-mt
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#
# The two main experiments. Every relation renders every camera of a scene, over all nine scene
# types. Submit from the repository root:
#
#   sbatch soc-cluster.sh               # both
#   sbatch soc-cluster.sh preliminary   # 20 small scenes per scene type (180 scenes), a few minutes
#   sbatch soc-cluster.sh large         # 100,000 randomized scenes, about 100 minutes on one H100
#
# The first gsplat call compiles its CUDA extension (about 12 minutes, once per environment).
# Results are written to results/. The --gres line requests an H100 on the NUS SoC cluster;
# change it for another cluster.

source ~/.bashrc
conda activate gsplat
pip install -r requirements.txt

STEPS=${*:-preliminary large}

for STEP in $STEPS; do
    case $STEP in
    preliminary)
        echo "=== preliminary study: results/preliminary.csv"
        python runners/run_preliminary.py --n-scenes 20 --out results/preliminary.csv
        ;;
    large)
        echo "=== large-scale study: results/allcam_100000.csv"
        python runners/run_large_scale.py --n-scenes 100000 --seed 2000 --out results/allcam_100000.csv
        ;;
    *)
        echo "unknown step '$STEP' (expected: preliminary, large)" >&2
        exit 1
        ;;
    esac
done
