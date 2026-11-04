"""
Where does a training step's time actually go?

Runs a short burst of real training steps under torch.profiler and reports the
time split three ways: compute, communication (NCCL all-reduce), and the input
pipeline (waiting on the DataLoader). That three-way breakdown is the figure
Philip's section is built around.

Launch it exactly like train_asl.py, at each GPU count you care about:

    srun -n 4 --cpus-per-task 32 --gpus-per-node 4 ./run_ddp.sh profile_step.py \\
        --splits_dir splits_block --data_root "$SCRATCH/asl/..." \\
        --model mobilenet_v2 --batch_size 256 --workers 8 --precision fp32 \\
        --out_dir prof-cpu-mobilenet_v2-4gpu

`bash slurm/week11_13.sh profile` submits these for both models at 1 and 4 GPUs.

It writes:
  <out_dir>/breakdown.json    the three-way split, for plotting
  <out_dir>/trace.json        a Chrome trace -- open at chrome://tracing
  and prints the usual profiler table to stdout.

The DataLoader wait is the number to watch. If it is large, the GPUs are
sitting idle waiting for JPEGs to be decoded, and no amount of communication
tuning will help until that is fixed.
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Reuse the real dataset and model definitions so the numbers mean something.
from train_asl import ListDataset, build_transforms, build_model


def classify(name):
    """Bucket a profiler event name into compute / comms / data."""
    n = name.lower()
    if any(k in n for k in ("nccl", "allreduce", "all_reduce", "c10d",
                            "broadcast", "reducer")):
        return "communication"
    if any(k in n for k in ("dataloader", "enumerate", "next", "fetch",
                            "collate", "to_copy", "pin_memory", "memcpyhtod")):
        return "data"
    return "compute"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--model", default="mobilenet_v2")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--bucket_mb", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out_dir", default="prof")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size > 1
    master = rank == 0

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if ddp:
        dist.init_process_group(backend=backend)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": None}[args.precision]
    if device.type == "cpu":
        amp_dtype = None

    meta = json.load(open(os.path.join(args.splits_dir, "meta.json")))
    data_root = args.data_root or meta["data_root"]
    if not os.path.isdir(data_root):
        raise SystemExit(f"data_root does not exist: {data_root}")

    train_tf, _ = build_transforms(args.img_size, "heavy")
    ds = ListDataset(data_root, os.path.join(args.splits_dir, "train.txt"), train_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=args.workers > 0)

    model = build_model(args.model, meta["num_classes"]).to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    bucket_cap_mb=args.bucket_mb)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    if master:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"profiling {args.model} | world_size {world_size} | "
              f"per-GPU batch {args.batch_size} | precision {args.precision}")

    # ---- a hand-timed data-wait measurement, which the profiler buckets poorly ----
    it = iter(loader)
    data_wait = 0.0
    step_total = 0.0

    def one_step(measure=False):
        nonlocal it, data_wait, step_total
        t0 = time.perf_counter()
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        if measure:
            data_wait += (t1 - t0)
            step_total += (t2 - t0)

    for _ in range(args.warmup):
        one_step()

    acts = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=acts, record_shapes=False,
                                with_stack=False) as prof:
        for _ in range(args.steps):
            one_step(measure=True)
            prof.step()

    if not master:
        if ddp:
            dist.destroy_process_group()
        return

    # ---- bucket the profiler events ----
    buckets = {"compute": 0.0, "communication": 0.0, "data": 0.0}
    key_avgs = prof.key_averages()
    for e in key_avgs:
        dev_us = getattr(e, "self_device_time_total", 0) or 0
        cpu_us = getattr(e, "self_cpu_time_total", 0) or 0
        buckets[classify(e.key)] += (dev_us if dev_us else cpu_us) / 1e6

    total_prof = sum(buckets.values()) or 1.0
    mean_step_ms = step_total / args.steps * 1000
    mean_data_ms = data_wait / args.steps * 1000

    print("\n" + "=" * 68)
    print(f"wall-clock mean step time : {mean_step_ms:8.2f} ms")
    print(f"  of which waiting on data: {mean_data_ms:8.2f} ms "
          f"({mean_data_ms/max(1e-9, mean_step_ms)*100:.1f}%)")
    print("\nprofiler event time, bucketed:")
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14s} {v:8.3f} s   {v/total_prof*100:5.1f}%")
    print("=" * 68)

    print("\n" + prof.key_averages().table(
        sort_by="self_device_time_total" if device.type == "cuda"
        else "self_cpu_time_total", row_limit=18))

    out = {
        "model": args.model, "world_size": world_size,
        "per_gpu_batch": args.batch_size, "precision": args.precision,
        "bucket_mb": args.bucket_mb, "workers": args.workers,
        "steps": args.steps,
        "mean_step_ms": mean_step_ms,
        "mean_data_wait_ms": mean_data_ms,
        "data_wait_fraction": mean_data_ms / max(1e-9, mean_step_ms),
        "profiler_bucket_seconds": buckets,
        "profiler_bucket_fraction": {k: v / total_prof for k, v in buckets.items()},
    }
    with open(os.path.join(args.out_dir, "breakdown.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    try:
        prof.export_chrome_trace(os.path.join(args.out_dir, "trace.json"))
    except Exception as e:
        print(f"(chrome trace export failed: {e})")
    print(f"\nwrote {args.out_dir}/breakdown.json and trace.json")
    print("Open trace.json at chrome://tracing to see the timeline.")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
