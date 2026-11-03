#!/bin/bash
# Generic batch job for one training configuration.
# Submit it with sbatch, overriding node/task counts and passing settings
# through --export. Nothing is hard-coded, so one file covers every run.
#
#   sbatch --nodes=1 --ntasks=4 \
#          --export=ALL,MODEL=mobilenet_v2,NGPU=4,PER_GPU_BATCH=64,EPOCHS=20,TAG=strong \
#          slurm/run_training.sh
#
# Use slurm/submit_sweep.sh to fire off the whole sweep at once.

#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --account=m4431_g
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --job-name=asl-train
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

# ---- settings, all overridable via --export ----
MODEL=${MODEL:-mobilenet_v2}
NGPU=${NGPU:-$SLURM_NTASKS}
PER_GPU_BATCH=${PER_GPU_BATCH:-128}
EPOCHS=${EPOCHS:-20}
TARGET_ACC=${TARGET_ACC:-0.95}
SPLITS=${SPLITS:-splits_block}
SCALE_LR=${SCALE_LR:-0}          # 1 = weak scaling (linear LR scaling rule)
TAG=${TAG:-run}
DATA_ROOT=${DATA_ROOT:-$SCRATCH/asl/asl_alphabet_train/asl_alphabet_train}
EXTRA=${EXTRA:-}

module load python/3.12-26.1.0
source "$SCRATCH/perlmutter-class/dist_dl/bin/activate"

OUT="out-${TAG}-${MODEL}-${NGPU}gpu-b${PER_GPU_BATCH}"
LOG="logs/${TAG}_${MODEL}_${NGPU}gpu_b${PER_GPU_BATCH}.log"
mkdir -p logs

SCALE_FLAG=""
if [ "$SCALE_LR" = "1" ]; then SCALE_FLAG="--scale_lr --warmup_epochs 2"; fi

echo "=== $(date) ==="
echo "job $SLURM_JOB_ID on $SLURM_JOB_NUM_NODES node(s), $NGPU GPU(s)"
echo "model $MODEL | per-GPU batch $PER_GPU_BATCH | global batch $((PER_GPU_BATCH*NGPU))"
echo "scale_lr $SCALE_LR | epochs $EPOCHS | out $OUT"

START=$(date +%s)
srun -n "$NGPU" --cpus-per-task 16 --gpus-per-node 4 ./run_ddp.sh train_asl.py \
    --splits_dir "$SPLITS" \
    --data_root "$DATA_ROOT" \
    --model "$MODEL" \
    --batch_size "$PER_GPU_BATCH" \
    --epochs "$EPOCHS" \
    --target_acc "$TARGET_ACC" \
    --out_dir "$OUT" \
    $SCALE_FLAG $EXTRA 2>&1 | tee "$LOG"
END=$(date +%s)

echo "WALLCLOCK_SECONDS $((END-START))" | tee -a "$LOG"
echo "=== done $(date) ==="
