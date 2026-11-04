# Scaling a Small Model on Perlmutter

ASL fingerspelling recognition as the workload for a distributed-training
scaling study. The question is not "can we classify handshapes" — one A100
does that comfortably — it is **where does data-parallel scaling stop paying
off for a small vision model, and why.**

Team: Vincent Chen, Rohan Manjarekar, Philip Guan, Tarun Sathyanarayana

**Start with `docs/ASL_Project_Perlmutter_Runbook.pdf`.** It is the plan: every
command, who owns what, who is waiting on whom, and the Week 10 checklist. Do
Part 0, then find your section.

`docs/ASL_Project_Team_Handbook.pdf` is the older overview: background, ground
rules (§2) and minimum versions (§7). If the two disagree, follow the runbook
and [the corrections below](#where-the-documents-differ).

| Person | Workstream | Runbook section |
|---|---|---|
| Philip | Splits, GPU-hour accounting, profiling and the pipeline fix | 1 |
| Vincent | Single-GPU baselines, target accuracy, `scratch_cnn`, webcam demo | 2 |
| Rohan | Weak and strong scaling, bandwidth, figures | 3 |
| Tarun | Large-batch study, evaluation, own-hands set, webcam demo | 4 |

---

## Quick start

Run all of this on Perlmutter.

```bash
# 1. environment
module load python/3.12-26.1.0
source "$SCRATCH/perlmutter-class/dist_dl/bin/activate"
pip install torchvision pillow matplotlib kaggle nvidia-ml-py

# 2. the repo
mkdir -p "$SCRATCH/asl" && cd "$SCRATCH/asl"
git clone https://github.com/rohandm285/asl-scaling-project.git repo
chmod +x repo/run_ddp.sh repo/slurm/*.sh

# 3. your own copy of the images (about 1 GB)
kaggle datasets download -d grassknoted/asl-alphabet && unzip -q asl-alphabet.zip

# 4. smoke test -- confirms your setup works
salloc --nodes 1 --qos shared_interactive --time 00:30:00 --constraint gpu \
       --account=m4431_g --ntasks 1 --gpus-per-node 1 --cpus-per-task 32

module load python/3.12-26.1.0
source "$SCRATCH/perlmutter-class/dist_dl/bin/activate"
cd "$SCRATCH/asl/repo"
srun -n 1 --cpus-per-task 32 --gpus-per-node 1 ./run_ddp.sh train_asl.py \
  --splits_dir splits_block \
  --data_root "$SCRATCH/asl/asl_alphabet_train/asl_alphabet_train" \
  --model mobilenet_v2 --batch_size 256 --precision fp32 --workers 8 \
  --warmup_steps 10 --timed_steps 30 --gpu_util --out_dir out-smoke

exit                       # release the GPU
squeue -u "$USER"          # should be empty
```

If that prints throughput, median step time, peak GPU memory and GPU
utilization, you are ready.

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
                       --out_dir splits_block --seed 0
python prepare_data.py --data_root "$SCRATCH/asl/asl_alphabet_train/asl_alphabet_train" \
                       --out_dir splits_random --mode random --seed 0
```

---

## Files

| File | What it does | Owner |
|---|---|---|
| `prepare_data.py` | Build leak-aware train/val/test file lists | Philip (once) |
| `train_asl.py` | DDP training and timed runs. Logs throughput, timing, accuracy, time-to-target | Vincent (everyone runs it; rule 1 below still applies) |
| `eval_asl.py` | Test-split and own-hands accuracy, per-class, confusion matrix | Tarun |
| `plot_results.py` | Scaling figures, plus `summary.csv` | Rohan |
| `nccl_bench.py` | Measured all-reduce bandwidth, for the prediction | Rohan |
| `profile_step.py` | Three-way time split: compute / comms / data loading | Philip |
| `demo.py` | Live webcam demo, runs on a laptop CPU | Vincent, Tarun |
| `gpu_pipeline.py` | GPU-resident dataset + GPU augmentation: the input-pipeline fix | Philip |
| `lars.py` | LARS optimizer, for the large-batch comparison | Tarun |
| `capture_hands.py` | Webcam capture of our own held-out test set | everyone |
| `plot_profile.py` | Stacked time breakdown and data-wait share | Philip |
| `run_ddp.sh` | Slurm → torch.distributed rank translation | — |
| `slurm/week10.sh` | Week 10: baselines + weak-scaling sweep, deck spec exactly | Vincent (baselines), Rohan (weak) |
| `slurm/week11_13.sh` | Weeks 11-13: profiling, large-batch, strong scaling | Philip, Tarun, Rohan |
| `slurm/gpu_hours.sh` | GPU-hour accounting from sacct, for the budget slide | Philip |
| `slurm/submit_sweep.sh` | Optional 20-epoch runs, see below. Not part of the planned sweep | Rohan |
| `slurm/run_training.sh` | One batch job for `submit_sweep.sh` | — |

---

## Running the sweep

Interactive runs are for debugging. Real runs go through the batch queue so you
are not babysitting a terminal:

| Command | What it submits | Owner |
|---|---|---|
| `bash slurm/week10.sh baselines` | Both models, 1 GPU, 20 epochs, batch 256, fp32, target 95% | Vincent |
| `bash slurm/week10.sh weak` | Timed: both models, 1 → 16 GPUs, 256 img/GPU, 50+300 steps | Rohan |
| `bash slurm/week11_13.sh profile` | Both models, 1 and 4 GPUs, batch 256 per GPU, CPU loader | Philip |
| `bash slurm/week11_13.sh strong` | Timed: both models, 1 → 16 GPUs, global batch 1024, 3 repeats | Rohan |
| `bash slurm/week11_13.sh batch` | MobileNetV2 on 4 GPUs, 20 epochs, global batch 256 / 1024 / 4096 × plain / linear scaling / LARS | Tarun |

```bash
squeue -u "$USER"                    # watch
grep RESULT logs/*.log               # the headline numbers
```

Then:

```bash
python plot_results.py --runs_dir . --out_dir figures \
    --bandwidth_json nccl_results.json
python plot_profile.py --runs_dir . --out_dir figures
```

**Timed runs measure throughput only.** They never evaluate, so they record no
accuracy, no time-to-target and no `best.pt`. Checkpoints come from the baselines
and the large-batch runs.

**`slurm/submit_sweep.sh` is a different experiment.** It runs 20-epoch training
in bf16, with weak scaling at 128 img/GPU and LR scaling, and strong scaling at a
global batch of 256. It is the only script that records time-to-target across GPU
counts. It does not pass `--tag`, so `plot_results.py` files its runs under
`weak` or `strong`: keep them out of the `--runs_dir` you use for the timed
sweep, or the two sets end up on the same curves.

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
python eval_asl.py --ckpt out-baseline-mobilenet_v2/best.pt \
    --splits_dir splits_block --split test \
    --data_root "$SCRATCH/asl/asl_alphabet_train/asl_alphabet_train"

# the same checkpoint on the naive split, for the leakage comparison
python eval_asl.py --ckpt out-baseline-mobilenet_v2/best.pt \
    --splits_dir splits_random --split test \
    --data_root "$SCRATCH/asl/asl_alphabet_train/asl_alphabet_train" --tag random

# our own hands (folder with one subfolder per class)
python eval_asl.py --ckpt out-baseline-mobilenet_v2/best.pt --image_dir ourhands
```

---

## The own-hands set

```bash
# on each person's laptop, not on Perlmutter
pip install opencv-python mediapipe
python capture_hands.py --person <your_name> --out ourhands
```

SPACE captures, `s` skips, `b` goes back, `q` quits.

---

## The demo

```bash
# on your laptop, in a terminal that is NOT inside the ssh session
scp <user>@perlmutter.nersc.gov:/pscratch/sd/<x>/<user>/asl/repo/out-baseline-mobilenet_v2/best.pt ~/Desktop/
pip install torch torchvision opencv-python mediapipe
python demo.py --ckpt ~/Desktop/best.pt
```

SPACE clears the spelled text, `q` quits.

---

## Ground rules

1. **Nobody edits `train_asl.py`.** Add a flag, or put it in your own file.
2. **Your own `--out_dir` and log file**, named after you, for anything you run
   by hand. The `slurm/week*.sh` scripts use fixed names (`out-baseline-*`,
   `out-weak-*`, ...) inside your own clone, so leave those as they are.
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
--optimizer adamw                                     # default
--optimizer lars --lr 4.0 --scale_lr --warmup_epochs 2  # layer-wise adaptive

# label a run so plot_results.py groups it into the right sweep
--tag baseline | weak | weakfixed | strong | batchplain | batchscaled | batchlars
```

## The three Week 10 commands

```bash
bash slurm/week10.sh baselines     # Vincent: single-GPU, both models, fp32, target 95%
bash slurm/week10.sh weak          # Rohan: 1 -> 16 GPUs, 256 img/GPU, 50+300 steps
bash slurm/gpu_hours.sh 2026-11-01 # Philip: measured GPU-hours by experiment
python plot_results.py --runs_dir . --out_dir figures
```

---

## Where the documents differ

Both PDFs open with a corrections page listing these. The handbook predates the
runbook and the scripts.

| Topic | The PDF says | Follow this |
|---|---|---|
| Baselines | Handbook: batch 128, default precision, `out-vincent-mnv2` | `bash slurm/week10.sh baselines`: batch 256, fp32 |
| Smoke test | Handbook: `interactive` QOS, batch 128, `--max_steps 50` | The Quick start above |
| Splits | Handbook: Rohan builds them under `$SCRATCH` | Philip built them; use the committed ones |
| Weak scaling | Handbook: `submit_sweep.sh weak`, `--scale_lr` | `bash slurm/week10.sh weak`: 256 img/GPU, no LR scaling |
| Strong scaling | Handbook: `submit_sweep.sh strong`, global batch 256 | `bash slurm/week11_13.sh strong`: global batch 1024 |
| Large batch | Handbook: two recipes, LARS as a stretch, `submit_sweep.sh batch` | `bash slurm/week11_13.sh batch`: three recipes, nine jobs |
| Profiling | Handbook: every GPU count | `bash slurm/week11_13.sh profile` covers 1 and 4 GPUs; add others by hand |
| Demo owner | Handbook: Rohan | Vincent and Tarun |
| Own-hands set | Handbook: photos, read by `prepare_data.py` | `capture_hands.py`, then `eval_asl.py --image_dir ourhands` |
| GPU-resident memory | Both: about 13 GB at 224x224; runbook: 2.6 GB per GPU on 4 GPUs | The training set is stored at 200x200: 8.4 GB in total, about 2.1 GB per GPU on 4 GPUs |
| Before/after figure | Runbook: `plot_profile.py` makes it | `profile_step.py` has no `--gpu_resident`, so that figure is skipped. Compare `RESULT_TIMED` throughput from `train_asl.py` with and without the flag |
