#!/bin/bash
# Submit the whole scaling sweep as batch jobs, then walk away.
#
#   bash slurm/submit_sweep.sh strong     # fixed global batch of 256
#   bash slurm/submit_sweep.sh weak       # fixed per-GPU batch of 128
#   bash slurm/submit_sweep.sh batch      # the large-batch accuracy study
#   bash slurm/submit_sweep.sh all
#
# Check on them with:  squeue -u "$USER"
# Cancel everything with: scancel -u "$USER"

set -euo pipefail
MODE=${1:-strong}
EPOCHS=${EPOCHS:-20}
TARGET_ACC=${TARGET_ACC:-0.95}

# GPU count -> nodes. Perlmutter has 4 GPUs per node.
nodes_for() {
    local n=$1
    if   [ "$n" -le 4 ];  then echo 1
    elif [ "$n" -le 8 ];  then echo 2
    elif [ "$n" -le 16 ]; then echo 4
    else echo $(( (n + 3) / 4 )); fi
}

submit() {  # submit <tag> <model> <ngpu> <per_gpu_batch> <scale_lr>
    local tag=$1 model=$2 ngpu=$3 batch=$4 scale=$5
    local nodes; nodes=$(nodes_for "$ngpu")
    local tasks_per_node=$(( ngpu < 4 ? ngpu : 4 ))
    echo "submit: $tag $model ${ngpu}gpu (${nodes} node) per-GPU batch $batch scale_lr=$scale"
    sbatch --nodes="$nodes" --ntasks="$ngpu" --ntasks-per-node="$tasks_per_node" \
        --job-name="asl-${tag}-${model}-${ngpu}" \
        --export=ALL,MODEL="$model",NGPU="$ngpu",PER_GPU_BATCH="$batch",\
EPOCHS="$EPOCHS",TARGET_ACC="$TARGET_ACC",SCALE_LR="$scale",TAG="$tag" \
        slurm/run_training.sh
}

run_strong() {
    # Global batch pinned at 256; each GPU gets a smaller slice as N grows.
    for model in mobilenet_v2 resnet18; do
        for n in 1 2 4 8 16; do
            submit strong "$model" "$n" $(( 256 / n )) 0
        done
    done
}

run_weak() {
    # Per-GPU batch pinned at 128; global batch grows with N, LR scales with it.
    for model in mobilenet_v2 resnet18; do
        for n in 1 2 4 8 16; do
            submit weak "$model" "$n" 128 1
        done
    done
}

run_batch() {
    # Large-batch accuracy study on a fixed 4 GPUs, with and without LR scaling.
    for gb in 256 1024 4096; do
        submit batchnoscale mobilenet_v2 4 $(( gb / 4 )) 0
        submit batchscaled  mobilenet_v2 4 $(( gb / 4 )) 1
    done
}

case "$MODE" in
    strong) run_strong ;;
    weak)   run_weak ;;
    batch)  run_batch ;;
    all)    run_strong; run_weak; run_batch ;;
    *) echo "usage: $0 [strong|weak|batch|all]"; exit 1 ;;
esac

echo
echo "submitted. watch with:  squeue -u \"\$USER\""
echo "when they finish:       python plot_results.py --runs_dir . --out_dir figures"
