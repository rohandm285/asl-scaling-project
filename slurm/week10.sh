#!/bin/bash
# Week 10 deliverables, exactly as the midterm deck specifies them.
#
#   bash slurm/week10.sh baselines   # Vincent: single-GPU baselines, both models, fp32
#   bash slurm/week10.sh weak        # Rohan: weak-scaling sweep, 1 -> 16 GPUs
#   bash slurm/week10.sh all
#
# Deck spec this follows:
#   baselines  fp32, batch 256, AdamW, 224x224, 8 workers, one GPU, shared QOS
#              reports val/test accuracy, img/s, time per epoch,
#              epochs to 95% val accuracy, peak GPU memory
#   weak       256 images per GPU at every point, 50 warm-up + 300 timed steps
#              1 and 2 GPUs on shared QOS, 4 and up on regular
#              reports images/sec and parallel efficiency
#
# Watch with:  squeue -u "$USER"

set -euo pipefail
MODE=${1:-all}
DATA_ROOT=${DATA_ROOT:-$SCRATCH/asl/asl_alphabet_train/asl_alphabet_train}
SPLITS=${SPLITS:-splits_block}
ACCOUNT=${ACCOUNT:-m4431_g}
mkdir -p logs

# 1-2 GPU jobs go to the shared QOS and are charged per GPU (0.25 node-hours
# per GPU-hour) instead of for a whole node.
qos_for()  { [ "$1" -le 2 ] && echo shared || echo regular; }
nodes_for() {
    if   [ "$1" -le 4 ];  then echo 1
    elif [ "$1" -le 8 ];  then echo 2
    else echo 4; fi
}

submit_baseline() {   # submit_baseline <model>
    local model=$1
    echo "baseline: $model  (1 GPU, shared QOS, fp32, batch 256)"
    sbatch --nodes=1 --ntasks=1 --ntasks-per-node=1 \
        --qos=shared --constraint=gpu --account="$ACCOUNT" \
        --gpus-per-node=1 --cpus-per-task=32 --time=01:30:00 \
        --job-name="asl-baseline-${model}" \
        --output="logs/slurm-baseline-${model}-%j.out" \
        --wrap="module load python/3.12-26.1.0 && \
source \$SCRATCH/perlmutter-class/dist_dl/bin/activate && \
srun -n 1 --cpus-per-task 32 --gpus-per-node 1 ./run_ddp.sh train_asl.py \
  --splits_dir $SPLITS --data_root '$DATA_ROOT' --model $model \
  --batch_size 256 --epochs 20 --precision fp32 --workers 8 \
  --target_acc 0.95 --gpu_util --tag baseline \
  --out_dir out-baseline-${model} 2>&1 | tee logs/baseline_${model}.log"
}

submit_weak() {       # submit_weak <model> <ngpu>
    local model=$1 ngpu=$2
    local nodes; nodes=$(nodes_for "$ngpu")
    local qos;   qos=$(qos_for "$ngpu")
    local tpn=$(( ngpu < 4 ? ngpu : 4 ))
    local gpn=$(( ngpu < 4 ? ngpu : 4 ))
    echo "weak: $model ${ngpu} GPU(s), ${nodes} node(s), ${qos} QOS, 256 img/GPU"
    sbatch --nodes="$nodes" --ntasks="$ngpu" --ntasks-per-node="$tpn" \
        --qos="$qos" --constraint=gpu --account="$ACCOUNT" \
        --gpus-per-node="$gpn" --cpus-per-task=32 --time=00:30:00 \
        --job-name="asl-weak-${model}-${ngpu}" \
        --output="logs/slurm-weak-${model}-${ngpu}-%j.out" \
        --wrap="module load python/3.12-26.1.0 && \
source \$SCRATCH/perlmutter-class/dist_dl/bin/activate && \
srun -n $ngpu --cpus-per-task 32 --gpus-per-node $gpn ./run_ddp.sh train_asl.py \
  --splits_dir $SPLITS --data_root '$DATA_ROOT' --model $model \
  --batch_size 256 --precision fp32 --workers 8 \
  --warmup_steps 50 --timed_steps 300 --gpu_util --tag weak \
  --out_dir out-weak-${model}-${ngpu}gpu 2>&1 | tee logs/weak_${model}_${ngpu}.log"
}

case "$MODE" in
    baselines)
        for m in mobilenet_v2 resnet18; do submit_baseline "$m"; done ;;
    weak)
        for m in mobilenet_v2 resnet18; do
            for n in 1 2 4 8 16; do submit_weak "$m" "$n"; done
        done ;;
    all)
        for m in mobilenet_v2 resnet18; do submit_baseline "$m"; done
        for m in mobilenet_v2 resnet18; do
            for n in 1 2 4 8 16; do submit_weak "$m" "$n"; done
        done ;;
    *) echo "usage: $0 [baselines|weak|all]"; exit 1 ;;
esac

cat <<'EOF'

submitted.

  squeue -u "$USER"                         watch the queue
  grep RESULT logs/*.log                    the headline numbers
  python plot_results.py --runs_dir . --out_dir figures
  bash slurm/gpu_hours.sh                   GPU-hour accounting
EOF
