#!/bin/bash
#SBATCH --job-name=gsplat-rootcause
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate gsplat

pip install -r requirements.txt

echo "=== 1. conic differences against the conditioning of the projected covariance"
python gsplat_mt/diagnose_conic_precision.py --n-scenes 200
python gsplat_mt/diagnose_conic_precision.py --gaussian 65923:308 --k 100

echo "=== 2. gsplat's rasterizer arithmetic, compiled, against torch"
python gsplat_mt/replicate_kernel_arithmetic.py --csv results/large_scale_w11.csv --csv-xl results/xl_scale_findings.csv

echo "=== 3. projection kernels in the built extension"
SO=~/.cache/torch_extensions/py312_cu130/gsplat_cuda/gsplat_cuda.so
cuobjdump -lelf $SO 2>&1 | head -5
cuobjdump -sass $SO 2>&1 | grep -iE "Function|fatal|error" | grep -iE "projection|fatal|error" | head -20
