#!/bin/bash
# Translates Slurm's rank variables into what torch.distributed expects.
# Works for any GPU count, on any number of nodes.
#
#   srun -n 4 --cpus-per-task 16 --gpus-per-node 4 ./run_ddp.sh train_asl.py --flags...
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
python "$@"
