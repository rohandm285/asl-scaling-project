# Scaling a Small Model on Perlmutter

ASL fingerspelling recognition as the workload for a distributed-training
scaling study. The question is not "can we classify handshapes" — one A100
does that comfortably — it is **where does data-parallel scaling stop paying
off for a small vision model, and why.**

Team: Vincent Chen, Rohan Manjarekar, Philip Guan, Tarun Sathyanarayana

Read `docs/ASL_Project_Team_Handbook.pdf` first. It has the full plan, who owns
what, and the ground rules.

---

## Quick start

```bash
# 1. environment
module load python/3.12-26.1.0
source "$SCRATCH/perlmutter-class/dist_dl/bin/activate"
pip install torchvision pillow matplotlib

# 2. your own copy of the images (about 1 GB)
mkdir -p "$SCRATCH/asl" && cd "$SCRATCH/asl"
kaggle datasets download -d grassknoted/asl-alphabet && unzip -q asl-alphabet.zip
cd -   # back to the repo

# 3. 50-step smoke test -- confirms your setup works
salloc --nodes 1 --qos interactive --time 00:30:00 --constraint gpu \
       --account=m4431_g --ntasks-per-node 4 --gpus-per-node 4 --cpus-per-task 16

module load python/3.12-26.1.0
source "$SCRATCH/perlmutter-class/dist_dl/bin/activate"
srun -n 1 --cpus-per-task 16 --gpus-per-node 1 ./run_ddp.sh train_asl.py \
  --splits_dir splits_block \
  --data_root "$SCRATCH/asl/asl_alphabet_train/asl_alphabet_train" \
  --model mobilenet_v2 --batch_size 128 --epochs 1 --max_steps 50 \
  --out_dir out-smoke
```

If that prints `step 20: ... img/s ...` lines, you are ready.

---

## The split files

`splits_block/` and `splits_random/` are committed to this repo. They are plain
text lists of `relative/path<TAB>label`, so everyone uses identical splits
regardless of whose `$SCRATCH` the images live on. **Always pass your own
`--data_root`.**

The two split modes exist because the ASL Alphabet images are consecutive video
frames. A random split puts near-identical frames on both sides and inflates
accuracy to ~99.9%, which is meaningless. `splits_block` splits by contiguous
frame index instead. Reporting both numbers is one of our results.

Rebuild them with:

```bash
python prepare_data.py --data_root "$SCRATCH/asl/asl_alphabet_train/asl_alphabet_train" \
                       --out_dir splits_block
python prepare_data.py --data_root "$SCRATCH/asl/asl_alphabet_train/asl_alphabet_train" \
                       --out_dir splits_random --mode random
```

---

## Files

| File | What it does | Owner |
|---|---|---|
| `prepare_data.py` | Build leak-aware train/val/test file lists | Rohan (once) |
| `train_asl.py` | DDP training. Logs throughput, timing, accuracy, time-to-target | everyone |
| `eval_asl.py` | Test-split and own-hands accuracy, per-class, confusion matrix | Tarun |
| `plot_results.py` | Every figure in the report, plus `summary.csv` | Rohan |
| `nccl_bench.py` | Measured all-reduce bandwidth, for the prediction | Rohan |
| `profile_step.py` | Three-way time split: compute / comms / data loading | Philip |
| `demo.py` | Live webcam demo, runs on a laptop CPU | Rohan |
| `gpu_pipeline.py` | GPU-resident dataset + GPU augmentation: the input-pipeline fix | Philip |
| `lars.py` | LARS optimizer, for the large-batch comparison | Tarun |
| `capture_hands.py` | Webcam capture of our own held-out test set | everyone |
| `plot_profile.py` | Stacked time breakdown + before/after pipeline figure | Philip |
| `run_ddp.sh` | Slurm → torch.distributed rank translation | — |
| `slurm/run_training.sh` | Generic batch job for one configuration | — |
| `slurm/submit_sweep.sh` | Fire off the whole sweep as batch jobs | Rohan |
| `slurm/week10.sh` | Week 10: baselines + weak-scaling sweep, deck spec exactly | Rohan |
| `slurm/week11_13.sh` | Weeks 11-13: profiling, large-batch, strong scaling | Philip, Tarun |
| `slurm/gpu_hours.sh` | GPU-hour accounting from sacct, for the budget slide | Philip |

---

## Running the sweep

Interactive runs are for debugging. Real runs go through the batch queue so you
are not babysitting a terminal:

```bash
bash slurm/submit_sweep.sh strong    # global batch fixed at 256
bash slurm/submit_sweep.sh weak      # per-GPU batch fixed, LR scaled
bash slurm/submit_sweep.sh batch     # the large-batch accuracy study

squeue -u "$USER"                    # watch
```

Then:

```bash
python plot_results.py --runs_dir . --out_dir figures \
    --bandwidth_json nccl_results.json
```

---

## Measuring bandwidth (needed for the prediction plot)

```bash
# inside one node
srun -n 4 --cpus-per-task 16 --gpus-per-node 4 ./run_ddp.sh nccl_bench.py --out nccl_intra.json
# across two nodes (allocate --nodes 2)
srun -n 8 --cpus-per-task 16 --gpus-per-node 4 ./run_ddp.sh nccl_bench.py --out nccl_inter.json
# merge
python nccl_bench.py --merge nccl_intra.json nccl_inter.json --out nccl_results.json
```

---

## Evaluation

```bash
# held-out test split, leak-aware
python eval_asl.py --ckpt out-strong-mobilenet_v2-1gpu-b256/best.pt \
    --splits_dir splits_block --split test \
    --data_root "$SCRATCH/asl/asl_alphabet_train/asl_alphabet_train"

# the same checkpoint on the naive split, for the leakage comparison
python eval_asl.py --ckpt ... --splits_dir splits_random --split test \
    --data_root ... --tag random

# our own hands (folder with one subfolder per class)
python eval_asl.py --ckpt ... --image_dir ourhands
```

---

## The demo

```bash
# on your laptop, in a terminal that is NOT inside the ssh session
scp <user>@perlmutter.nersc.gov:/pscratch/sd/<x>/<user>/.../best.pt ~/Desktop/
pip install torch torchvision opencv-python mediapipe
python demo.py --ckpt ~/Desktop/best.pt
```

SPACE clears the spelled text, `q` quits.

---

## Ground rules

1. **Nobody edits `train_asl.py`.** Add a flag, or put it in your own file.
2. **Your own `--out_dir` and log file**, named after you.
3. **Commit logs and `history.json`**, not just conclusions.
4. **Weekly 15-minute check-in**: what did you *push*. Two weeks of nothing and
   that workstream drops to its minimum version (handbook §7).

## Gotchas

- `scp` must be typed on the machine you want files to land on, in a terminal
  that is not inside the ssh session.
- `module load` and the venv activate do **not** survive into a new `salloc`
  shell. Run them again every time.
- `exit` to release an allocation, then `squeue -u "$USER"` to confirm.
- Always pass `--data_root`. The path inside `meta.json` is from whoever built
  the splits.
- Put `time` in front of interactive `srun` calls. The batch scripts record
  `WALLCLOCK_SECONDS` for you.


---

## New flags worth knowing

```bash
# scaling-sweep mode: fixed work per GPU, warm up, time N steps, exit.
# This is what the weak sweep uses -- NOT a convergence run.
--warmup_steps 50 --timed_steps 300

# GPU utilization via NVML, sampled every 0.5 s over the timed region,
# plus peak GPU memory. Needs: pip install nvidia-ml-py
--gpu_util

# the input-pipeline fix: decode once into GPU memory, augment on the GPU.
# Run the same config with and without it; the difference is the answer.
--gpu_resident

# optimizers for the large-batch study
--optimizer adamw            # default
--optimizer lars --lr 4.0    # layer-wise adaptive, for global batch >= 1024

# label a run so plot_results.py groups it into the right sweep
--tag weak | strong | batchplain | batchscaled | batchlars
```

## The three Week 10 commands

```bash
bash slurm/week10.sh baselines     # single-GPU, both models, fp32, target 95%
bash slurm/week10.sh weak          # 1 -> 16 GPUs, 256 img/GPU, 50+300 steps
bash slurm/gpu_hours.sh 2026-11-01 # measured GPU-hours by experiment
python plot_results.py --runs_dir . --out_dir figures
```
