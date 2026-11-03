#!/bin/bash
# Weeks 11-13: the bottleneck study and the large-batch study.
#
#   bash slurm/week11_13.sh profile     # Philip: time breakdown, before and after the fix
#   bash slurm/week11_13.sh batch       # Tarun: accuracy vs global batch, 3 optimizers
#   bash slurm/week11_13.sh strong      # Rohan: strong scaling, 3 repeats
#
set -euo pipefail
MODE=${1:-profile}
DATA_ROOT=${DATA_ROOT:-$SCRATCH/asl/asl_alphabet_train/asl_alphabet_train}
SPLITS=${SPLITS:-splits_block}
ACCOUNT=${ACCOUNT:-m4431_g}
mkdir -p logs
qos_for() { [ "$1" -le 2 ] && echo shared || echo regular; }
nodes_for() { if [ "$1" -le 4 ]; then echo 1; elif [ "$1" -le 8 ]; then echo 2; else echo 4; fi; }

run() {  # run <jobname> <ngpu> <time> <python args...>
    local name=$1 ngpu=$2 tlim=$3; shift 3
    local nodes; nodes=$(nodes_for "$ngpu")
    local gpn=$(( ngpu < 4 ? ngpu : 4 ))
    echo "submit: $name (${ngpu} GPU)"
    sbatch --nodes="$nodes" --ntasks="$ngpu" --ntasks-per-node="$gpn" \
        --qos="$(qos_for "$ngpu")" --constraint=gpu --account="$ACCOUNT" \
        --gpus-per-node="$gpn" --cpus-per-task=32 --time="$tlim" \
        --job-name="$name" --output="logs/slurm-${name}-%j.out" \
        --wrap="module load python/3.12-26.1.0 && \
source \$SCRATCH/perlmutter-class/dist_dl/bin/activate && \
srun -n $ngpu --cpus-per-task 32 --gpus-per-node $gpn ./run_ddp.sh $* \
  2>&1 | tee logs/${name}.log"
}

case "$MODE" in
  profile)
    # The same configuration twice: CPU loader, then the GPU-resident fix.
    for m in mobilenet_v2 resnet18; do
      for n in 1 4; do
        run "asl-prof-cpu-${m}-${n}" "$n" 00:40:00 profile_step.py \
          --splits_dir "$SPLITS" --data_root "'$DATA_ROOT'" --model "$m" \
          --batch_size 256 --workers 8 --precision fp32 \
          --out_dir "prof-cpu-${m}-${n}gpu"
      done
    done
    echo "Run the GPU-resident half after confirming the shard fits:"
    echo "  profile_step.py does not take --gpu_resident; measure that side with"
    echo "  train_asl.py --gpu_resident --timed_steps 300 and compare throughput."
    ;;
  batch)
    # Accuracy vs global batch, three optimizer recipes.
    for gb in 256 1024 4096; do
      run "asl-batch-plain-${gb}"  4 01:30:00 train_asl.py \
        --splits_dir "$SPLITS" --data_root "'$DATA_ROOT'" --model mobilenet_v2 \
        --batch_size $((gb/4)) --epochs 20 --precision fp32 --workers 8 \
        --target_acc 0.95 --optimizer adamw --tag batchplain \
        --out_dir "out-batch-plain-${gb}"
      run "asl-batch-scaled-${gb}" 4 01:30:00 train_asl.py \
        --splits_dir "$SPLITS" --data_root "'$DATA_ROOT'" --model mobilenet_v2 \
        --batch_size $((gb/4)) --epochs 20 --precision fp32 --workers 8 \
        --target_acc 0.95 --optimizer adamw --scale_lr --warmup_epochs 2 \
        --tag batchscaled --out_dir "out-batch-scaled-${gb}"
      run "asl-batch-lars-${gb}"   4 01:30:00 train_asl.py \
        --splits_dir "$SPLITS" --data_root "'$DATA_ROOT'" --model mobilenet_v2 \
        --batch_size $((gb/4)) --epochs 20 --precision fp32 --workers 8 \
        --target_acc 0.95 --optimizer lars --lr 4.0 --scale_lr --warmup_epochs 2 \
        --tag batchlars --out_dir "out-batch-lars-${gb}"
    done
    ;;
  strong)
    for rep in 1 2 3; do
      for m in mobilenet_v2 resnet18; do
        for n in 1 2 4 8 16; do
          run "asl-strong-${m}-${n}-r${rep}" "$n" 00:30:00 train_asl.py \
            --splits_dir "$SPLITS" --data_root "'$DATA_ROOT'" --model "$m" \
            --batch_size $((1024/n)) --precision fp32 --workers 8 \
            --warmup_steps 50 --timed_steps 300 --gpu_util --tag strong \
            --out_dir "out-strong-${m}-${n}gpu-r${rep}"
        done
      done
    done
    ;;
  *) echo "usage: $0 [profile|batch|strong]"; exit 1 ;;
esac
echo; echo "squeue -u \"\$USER\""
