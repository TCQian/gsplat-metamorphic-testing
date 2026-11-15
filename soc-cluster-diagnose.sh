#!/bin/bash
#SBATCH --job-name=gsplat-mt-diagnose
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --mail-type=ALL
#
# Diagnoses of the large-scale run of soc-cluster.sh. Run that first, then pick one or more steps:
#
#   sbatch soc-cluster-diagnose.sh violations
#   sbatch soc-cluster-diagnose.sh undetected followup
#
# Steps on the large-scale run (results/allcam_100000.csv; set CSV=... to use another
# run_large_scale.py CSV made with the default seed and scene types):
#   violations   bisect every MR1/MR4 `violation` at the camera that shows it, then account for
#                every over-tolerance pixel of every camera and inexact factor of those scenes
#                -> results/open_violations.json, results/open_pixels.json
#   undetected   replay every scene where a planted fault was missed and classify the cause
#                -> results/undetected_allcam_100000.csv
#   followup     needs `undetected`: re-run the missed shuffle faults with a shuffle that moves
#                every Gaussian (derangement), check whether the affected Gaussians reach any pixel
#                (nothing), and test why dense_cover covers about 45% of pixels (coverage)
#                -> results/open_{derangement,nothing,coverage}.json
#   mr4          bisect every MR4 `tie_ambiguous` and `isolated_discontinuity` scene at the camera
#                that shows it -> results/bisect/mr4_{tie_ambiguous,isolated_discontinuity}.jsonl
#   mr1          the same for every MR1 `isolated_discontinuity` -> results/bisect/mr1_isolated_discontinuity.jsonl
#                mr4 and mr1 take about 2 s per scene (several GPU-hours for a 100,000-scene run).
#                PART=j/K runs every K-th scene from the j-th (j = 0..K-1) into its own file, so K
#                jobs cover all scenes; a rerun resumes a file. Pattern counts over the files:
#                python diagnosis/diagnose_mr4.py --summarize results/bisect/*.jsonl
# Standalone:
#   ties         how gsplat orders Gaussians with equal or one-float-step-apart depths
#   conic        conic rounding against the condition number of the projected covariance
#                (replays the earlier camera-0 run, seed 1000)
#   stage        which projection stage first differs between the packed and unpacked kernels;
#                recompiles parts of gsplat's CUDA source (replays the earlier camera-0 run)
#
# Output goes to the job's slurm-<id>.out; the steps marked -> also write files. Other tools
# in diagnosis/ (diagnose_mr1_inputs, diagnose_open_items, replicate_kernel_arithmetic) take
# CSVs of the earlier camera-0 runs and are run directly; see their --help.

source ~/.bashrc
conda activate gsplat
pip install -r requirements.txt

CSV=${CSV:-results/allcam_100000.csv}
FINDINGS=${CSV%.csv}_findings.csv
UNDETECTED=results/undetected_$(basename "$CSV")
SPLIT="" SUFFIX=""
if [ -n "$PART" ]; then
    SPLIT="--offset ${PART%/*} --every ${PART#*/}"
    SUFFIX="_part${PART%/*}of${PART#*/}"
fi

if [ $# -eq 0 ]; then
    echo "no step given; see the comments at the top of soc-cluster-diagnose.sh" >&2
    exit 1
fi

for STEP in "$@"; do
    echo "=== $STEP"
    case $STEP in
    violations)
        python diagnosis/diagnose_allcam.py violations --csv "$CSV"
        python diagnosis/diagnose_allcam.py pixels --csv "$CSV"
        ;;
    undetected)
        # single-Gaussian scenes are trivially undetectable for the shuffle fault
        python diagnosis/study_undetected_faults.py --csv "$CSV" --skip-modes single --out "$UNDETECTED"
        ;;
    followup)
        python diagnosis/diagnose_allcam.py derangement --undetected "$UNDETECTED"
        python diagnosis/diagnose_allcam.py nothing --undetected "$UNDETECTED"
        python diagnosis/diagnose_allcam.py coverage --csv "$CSV" --n 60
        ;;
    mr4)
        for V in tie_ambiguous isolated_discontinuity; do
            python diagnosis/diagnose_mr4.py --scenes-from "$FINDINGS" --where mr4_verdict=$V $SPLIT \
                --out results/bisect/mr4_$V$SUFFIX.jsonl --resume
        done
        ;;
    mr1)
        python diagnosis/diagnose_mr1.py --scenes-from "$FINDINGS" --where mr1_verdict=isolated_discontinuity $SPLIT \
            --out results/bisect/mr1_isolated_discontinuity$SUFFIX.jsonl --resume
        ;;
    ties)
        python diagnosis/probe_ties.py
        ;;
    conic)
        python diagnosis/diagnose_conic_precision.py --n-scenes 200
        ;;
    stage)
        # two large translation units are compiled; more parallel nvcc runs do not fit in memory
        MAX_JOBS=1 python diagnosis/isolate_projection_stage.py --n-scenes 100 --identity-rot --covars \
            --pairing --three-term --three-term-full --stages 0 1 2 3 4
        ;;
    *)
        echo "unknown step '$STEP'" >&2
        ;;
    esac
done
