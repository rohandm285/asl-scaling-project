"""
Data-parallel training for ASL fingerspelling classification on Perlmutter.

Designed to be launched exactly like the nanoGPT runs from assignment 2:

    srun -n 4 --cpus-per-task 16 --gpus-per-node 4 ./run_ddp.sh train_asl.py \
        --model=mobilenet_v2 --batch_size=128 --epochs=20

run_ddp.sh exports RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT
from the Slurm variables; this script reads those. With none of them set it
runs single-process on whatever device is available, including CPU.

Every log line is machine-readable so the plotting script can parse it:

    step 120: loss 1.8214, lr 9.81e-04, img/s 4821.3, time 26.6ms
    epoch 3: train_loss 0.4120, val_loss 0.5510, val_acc 0.8734, epoch_time 41.2s
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import torchvision.transforms as T
from PIL import Image


# --------------------------------------------------------------------------
# Dataset: reads the file lists written by prepare_data.py
# --------------------------------------------------------------------------
class ListDataset(Dataset):
    def __init__(self, data_root, list_file, transform):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        with open(list_file) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                rel, label = line.split("\t")
                self.samples.append((rel, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rel, label = self.samples[i]
        img = Image.open(os.path.join(self.data_root, rel)).convert("RGB")
        return self.transform(img), label


def build_transforms(img_size, aug_strength):
    """Augmentation is the main defence against the model memorising backgrounds."""
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if aug_strength == "none":
        train_tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])
    elif aug_strength == "light":
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.ToTensor(), norm,
        ])
    else:  # "heavy" -- the default
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomRotation(15),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(), norm,
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])

    eval_tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])
    return train_tf, eval_tf


def build_model(name, num_classes):
    import torchvision.models as models
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "scratch_cnn":
        # Small baseline with no pretrained weights, for the "what does transfer
        # learning buy us" comparison.
        m = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes),
        )
    else:
        raise ValueError(f"unknown model {name}")
    return m


def lr_at(step, total_steps, warmup_steps, peak_lr, min_lr):
    """Linear warmup then cosine decay."""
    import math
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


class GpuUtilSampler:
    """
    Samples this rank's GPU utilization with NVML, the same counter nvidia-smi
    prints, on a background thread. Start it when timing starts, stop it when
    timing ends, so the mean covers only the measured region.

    NVML "utilization" is the fraction of time ANY kernel was running, not how
    full the SMs were. Low numbers mean the GPU was idle -- waiting on the data
    loader or on the all-reduce. It does not tell you the GPU was used well.
    """

    def __init__(self, device_index, interval=0.5):
        self.interval = interval
        self.samples = []
        self.mem_samples = []
        self._stop = None
        self._thread = None
        self._handle = None
        self._nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:
            print(f"(gpu util sampling unavailable: {e}); "
                  f"pip install nvidia-ml-py to enable", flush=True)

    @property
    def available(self):
        return self._handle is not None

    def start(self):
        if not self.available:
            return
        import threading
        self._stop = threading.Event()

        def loop():
            while not self._stop.wait(self.interval):
                try:
                    u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self.samples.append(u.gpu)
                    m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                    self.mem_samples.append(m.used / 1e9)
                except Exception:
                    break

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self):
        if not self.samples:
            return None
        s = sorted(self.samples)
        return {
            "n_samples": len(s),
            "mean_util_pct": sum(s) / len(s),
            "median_util_pct": s[len(s) // 2],
            "min_util_pct": s[0],
            "max_util_pct": s[-1],
            "mean_mem_used_GB": (sum(self.mem_samples) / len(self.mem_samples)
                                 if self.mem_samples else None),
        }


def run_timed(model, loader, opt, crit, device, amp_dtype, scaler, args,
              world_size, master, sampler):
    """
    Scaling-sweep mode. Fixed work per GPU, warm up, then time a fixed number of
    steps. Reports images per second across all ranks -- the number the parallel
    efficiency is computed from. No evaluation, no epochs.
    """
    model.train()
    it = iter(loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    def one_step():
        x, y = next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        return loss

    if master:
        print(f"warmup: {args.warmup_steps} steps", flush=True)
    for i in range(args.warmup_steps):
        one_step()
        if master and (i + 1) % 25 == 0:
            print(f"  warmup {i+1}/{args.warmup_steps}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    if world_size > 1:
        dist.barrier()

    sampler.start()
    if master:
        print(f"timing: {args.timed_steps} steps", flush=True)
    t0 = time.time()
    per_step = []
    for i in range(args.timed_steps):
        s0 = time.time()
        one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_step.append(time.time() - s0)
        if master and (i + 1) % 100 == 0:
            done = time.time() - t0
            print(f"  step {i+1}/{args.timed_steps}  "
                  f"{(i+1)*args.batch_size*world_size/done:.1f} img/s", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    if world_size > 1:
        dist.barrier()
    elapsed = time.time() - t0
    sampler.stop()

    images = args.timed_steps * args.batch_size * world_size
    ips = images / elapsed
    ps = sorted(per_step)
    return {
        "timed_steps": args.timed_steps,
        "warmup_steps": args.warmup_steps,
        "elapsed_s": elapsed,
        "images_processed": images,
        "throughput_img_s": ips,
        "median_step_ms": ps[len(ps) // 2] * 1000,
        "p90_step_ms": ps[int(len(ps) * 0.9)] * 1000,
        "peak_mem_GB": (torch.cuda.max_memory_allocated(device) / 1e9
                        if device.type == "cuda" else None),
        "gpu_util": sampler.summary(),
    }


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, world_size):
    model.eval()
    loss_sum = torch.zeros(1, device=device)
    correct = torch.zeros(1, device=device)
    total = torch.zeros(1, device=device)
    crit = nn.CrossEntropyLoss(reduction="sum")

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            out = model(x)
            loss = crit(out.float(), y)
        loss_sum += loss
        correct += (out.argmax(1) == y).sum()
        total += y.numel()

    if world_size > 1:
        for t in (loss_sum, correct, total):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    model.train()
    n = total.item()
    return (loss_sum.item() / max(1, n)), (correct.item() / max(1, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--data_root", default=None,
                    help="Override the data_root recorded in meta.json. Use this when "
                         "the split files were made on someone else's $SCRATCH.")
    ap.add_argument("--model", default="mobilenet_v2")
    ap.add_argument("--batch_size", type=int, default=128,
                    help="PER-GPU batch size")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=float, default=1.0)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--aug", choices=["none", "light", "heavy"], default="heavy")
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--scale_lr", action="store_true",
                    help="Linear scaling rule: multiply lr by world_size (Goyal et al.)")
    ap.add_argument("--bucket_mb", type=int, default=25,
                    help="DDP gradient bucket size in MB -- the knob to tune for comms")
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--out_dir", default="out-asl")
    ap.add_argument("--target_acc", type=float, default=0.0,
                    help="If >0, record wall-clock time when val acc first reaches this")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="Stop after N steps regardless of epochs (for quick tests)")
    # ---- timed-run mode: the scaling sweep, not a convergence run ----
    ap.add_argument("--timed_steps", type=int, default=0,
                    help="Scaling-sweep mode: after --warmup_steps, time exactly this "
                         "many steps, report throughput and exit. No epochs, no eval. "
                         "This is what the weak-scaling sweep uses.")
    ap.add_argument("--warmup_steps", type=int, default=50,
                    help="Untimed steps before timing starts (timed-run mode only)")
    ap.add_argument("--optimizer", choices=["adamw", "lars", "sgd"], default="adamw",
                    help="lars = layer-wise adaptive rates, for the large-batch study")
    ap.add_argument("--trust_coefficient", type=float, default=0.001,
                    help="LARS eta; 0.001 is the value from the paper")
    ap.add_argument("--gpu_resident", action="store_true",
                    help="Decode the dataset once into GPU memory and augment on the "
                         "GPU, bypassing the CPU input pipeline. The before/after "
                         "comparison for the bottleneck section.")
    ap.add_argument("--tag", default=None,
                    help="Label for this run, e.g. weak / strong / batch. Used by "
                         "plot_results.py to group runs into sweeps.")
    ap.add_argument("--gpu_util", action="store_true",
                    help="Sample this rank's GPU utilization via NVML every 0.5 s and "
                         "report the mean over the timed region")
    args = ap.parse_args()

    # ---------------- distributed setup ----------------
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

    # ---------------- data ----------------
    meta = json.load(open(os.path.join(args.splits_dir, "meta.json")))
    data_root = args.data_root or meta["data_root"]
    num_classes = meta["num_classes"]
    if not os.path.isdir(data_root):
        raise SystemExit(
            f"data_root does not exist: {data_root}\n"
            f"The split files were built on another account. Pass --data_root "
            f"pointing at YOUR copy of asl_alphabet_train."
        )

    train_tf, eval_tf = build_transforms(args.img_size, args.aug)
    train_ds = ListDataset(data_root, os.path.join(args.splits_dir, "train.txt"), train_tf)
    val_ds = ListDataset(data_root, os.path.join(args.splits_dir, "val.txt"), eval_tf)

    # Validation always goes through the normal CPU loader: it runs rarely and
    # is not what the bottleneck study is measuring.
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        persistent_workers=args.workers > 0,
    )

    if args.gpu_resident:
        # Philip's bottleneck fix: decode the training set once into GPU memory
        # and do augmentation there, so the CPU input pipeline is out of the
        # picture entirely. Shards across ranks, so each GPU holds only its own
        # slice.
        from gpu_pipeline import GpuResidentLoader
        train_sampler = None
        train_loader = GpuResidentLoader(
            data_root, os.path.join(args.splits_dir, "train.txt"),
            batch_size=args.batch_size, img_size=args.img_size, device=device,
            rank=rank, world_size=world_size, aug=args.aug, verbose=master)
    else:
        train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True, persistent_workers=args.workers > 0,
        )

    # ---------------- model ----------------
    model = build_model(args.model, num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    bucket_cap_mb=args.bucket_mb)

    peak_lr = args.lr * (world_size if args.scale_lr else 1)
    if args.optimizer == "lars":
        from lars import LARS, build_lars_param_groups
        raw_model = model.module if ddp else model
        opt = LARS(build_lars_param_groups(raw_model, args.weight_decay),
                   lr=peak_lr, momentum=0.9,
                   trust_coefficient=args.trust_coefficient)
    elif args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=peak_lr, momentum=0.9,
                              weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                                weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16"
                                                  and device.type == "cuda"))

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)
    global_batch = args.batch_size * world_size

    if master:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"model {args.model}: {n_params/1e6:.2f}M parameters, "
              f"gradient size {n_params*4/1e6:.1f} MB in fp32")
        print(f"world_size {world_size} | per-GPU batch {args.batch_size} | "
              f"global batch {global_batch}")
        print(f"optimizer {args.optimizer} | "
              f"input pipeline {'GPU-resident' if args.gpu_resident else 'CPU DataLoader'}")
        print(f"lr {peak_lr:.3e} (scale_lr={args.scale_lr}) | "
              f"{steps_per_epoch} steps/epoch | {total_steps} total steps")
        print(f"train {len(train_ds)} | val {len(val_ds)} | classes {num_classes}")

    # ---------------- timed-run mode: the scaling sweep ----------------
    sampler = GpuUtilSampler(local_rank) if (args.gpu_util and device.type == "cuda") \
        else GpuUtilSampler(-1)
    if args.timed_steps:
        res = run_timed(model, train_loader, opt, crit, device, amp_dtype,
                        scaler, args, world_size, master, sampler)
        if master:
            os.makedirs(args.out_dir, exist_ok=True)
            util = res["gpu_util"]
            print("\n" + "=" * 70)
            print(f"model            {args.model}")
            print(f"GPUs             {world_size}")
            print(f"per-GPU batch    {args.batch_size}")
            print(f"global batch     {global_batch}")
            print(f"THROUGHPUT       {res['throughput_img_s']:.1f} img/s")
            print(f"median step      {res['median_step_ms']:.2f} ms "
                  f"(p90 {res['p90_step_ms']:.2f} ms)")
            if res["peak_mem_GB"]:
                print(f"peak GPU memory  {res['peak_mem_GB']:.2f} GB")
            if util:
                print(f"GPU utilization  {util['mean_util_pct']:.1f}% mean, "
                      f"{util['median_util_pct']:.0f}% median "
                      f"({util['n_samples']} samples)")
            print("=" * 70)
            print(f"RESULT_TIMED model={args.model} world_size={world_size} "
                  f"per_gpu_batch={args.batch_size} global_batch={global_batch} "
                  f"throughput_img_s={res['throughput_img_s']:.1f} "
                  f"median_step_ms={res['median_step_ms']:.2f} "
                  f"peak_mem_GB={res['peak_mem_GB'] or -1:.2f} "
                  f"gpu_util_pct={util['mean_util_pct'] if util else -1:.1f}",
                  flush=True)
            with open(os.path.join(args.out_dir, "history.json"), "w") as fh:
                json.dump({"args": vars(args), "world_size": world_size,
                           "global_batch": global_batch, "n_params": n_params,
                           "grad_bytes_fp32": n_params * 4,
                           "mode": "timed",
                           "median_img_s": res["throughput_img_s"],
                           "timed": res, "history": []}, fh, indent=2)
        if ddp:
            dist.destroy_process_group()
        return

    # ---------------- training loop ----------------
    step = 0
    best_acc = 0.0
    run_start = time.time()
    time_to_target = None
    epochs_to_target = None
    history = []
    step_throughputs = []   # images/sec samples, for the scaling plots

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()
        running_loss, running_n = 0.0, 0
        window_start = time.time()
        window_imgs = 0

        for x, y in train_loader:
            lr = lr_at(step, total_steps, warmup_steps, peak_lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                out = model(x)
                loss = crit(out, y)

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            running_loss += loss.item() * y.numel()
            running_n += y.numel()
            window_imgs += y.numel() * world_size
            step += 1

            if master and step % args.log_interval == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - window_start
                ips = window_imgs / dt
                step_throughputs.append(ips)
                print(f"step {step}: loss {loss.item():.4f}, lr {lr:.3e}, "
                      f"img/s {ips:.1f}, time {dt/args.log_interval*1000:.1f}ms",
                      flush=True)
                window_start = time.time()
                window_imgs = 0

            if args.max_steps and step >= args.max_steps:
                break

        val_loss, val_acc = evaluate(model, val_loader, device, amp_dtype, world_size)
        epoch_time = time.time() - epoch_start
        train_loss = running_loss / max(1, running_n)

        if master:
            print(f"epoch {epoch+1}: train_loss {train_loss:.4f}, "
                  f"val_loss {val_loss:.4f}, val_acc {val_acc:.4f}, "
                  f"epoch_time {epoch_time:.1f}s", flush=True)
            history.append({"epoch": epoch + 1, "train_loss": train_loss,
                            "val_loss": val_loss, "val_acc": val_acc,
                            "epoch_time": epoch_time,
                            "elapsed": time.time() - run_start})

            if args.target_acc and time_to_target is None and val_acc >= args.target_acc:
                time_to_target = time.time() - run_start
                epochs_to_target = epoch + 1
                print(f"TIME_TO_ACC {args.target_acc}: {time_to_target:.1f}s "
                      f"after {epochs_to_target} epochs", flush=True)

            if val_acc > best_acc:
                best_acc = val_acc
                raw = model.module if ddp else model
                torch.save({
                    "model_state": raw.state_dict(),
                    "model_name": args.model,
                    "classes": meta["classes"],
                    "img_size": args.img_size,
                    "val_acc": val_acc,
                    "epoch": epoch + 1,
                }, os.path.join(args.out_dir, "best.pt"))

        if args.max_steps and step >= args.max_steps:
            break

    total_time = time.time() - run_start
    if master:
        # Median rather than mean: the first window includes warmup and the
        # windows straddling an eval pass are artificially slow.
        thr = sorted(step_throughputs[1:]) or [0.0]
        median_ips = thr[len(thr) // 2]

        peak_mem = (torch.cuda.max_memory_allocated(device) / 1e9
                    if device.type == "cuda" else None)
        print(f"RESULT model={args.model} world_size={world_size} "
              f"per_gpu_batch={args.batch_size} global_batch={global_batch} "
              f"median_img_s={median_ips:.1f} best_val_acc={best_acc:.4f} "
              f"total_time_s={total_time:.1f} "
              f"epochs_to_target={epochs_to_target if epochs_to_target else -1} "
              f"peak_mem_GB={peak_mem if peak_mem else -1:.2f} "
              f"time_to_target_s={time_to_target if time_to_target else -1}",
              flush=True)
        with open(os.path.join(args.out_dir, "history.json"), "w") as fh:
            json.dump({"args": vars(args), "world_size": world_size,
                       "global_batch": global_batch, "n_params": n_params,
                       "grad_bytes_fp32": n_params * 4,
                       "history": history, "best_val_acc": best_acc,
                       "total_time_s": total_time,
                       "median_img_s": median_ips,
                       "step_throughputs": step_throughputs,
                       "mode": "epochs",
                       "epochs_to_target": epochs_to_target,
                       "peak_mem_GB": peak_mem,
                       "mean_epoch_time_s": (sum(e["epoch_time"] for e in history)
                                             / max(1, len(history))),
                       "time_to_target_s": time_to_target}, fh, indent=2)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
