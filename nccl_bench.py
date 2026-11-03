"""
Measure the real all-reduce bandwidth of this allocation.

This is what turns the scaling section from "we measured a curve" into "we
predicted where it would break and then checked". Run it once inside a single
node and once across two nodes; the two numbers are what the prediction uses.

    # inside one node
    srun -n 4 --cpus-per-task 16 --gpus-per-node 4 ./run_ddp.sh nccl_bench.py \\
        --out nccl_intra.json

    # across two nodes (allocate --nodes 2 first)
    srun -n 8 --cpus-per-task 16 --gpus-per-node 4 ./run_ddp.sh nccl_bench.py \\
        --out nccl_inter.json

    # then merge them for the plotting script
    python nccl_bench.py --merge nccl_intra.json nccl_inter.json \\
        --out nccl_results.json

The reported number is ALGORITHMIC bandwidth: payload bytes divided by elapsed
time. Ring all-reduce actually moves 2*(N-1)/N times that much per rank, and
the script prints the bus bandwidth too. Use the bus number when reasoning
about the hardware, and the algorithmic number when predicting step time.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.distributed as dist


# Sizes chosen to bracket our two models' gradients:
# MobileNetV2 ~8.9 MB, ResNet-18 ~44.7 MB in fp32.
DEFAULT_SIZES_MB = [1, 4, 8.9, 16, 44.7, 64, 128, 256]


def bench_one(numel, device, iters, warmup):
    x = torch.ones(numel, dtype=torch.float32, device=device)
    for _ in range(warmup):
        dist.all_reduce(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dist.barrier()

    times = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="nccl_bench.json")
    ap.add_argument("--sizes_mb", type=float, nargs="*", default=None)
    ap.add_argument("--merge", nargs="*", default=None,
                    help="Merge previously written result files into one json "
                         "(run this without srun)")
    args = ap.parse_args()

    # ---- merge mode: no distributed setup needed ----
    if args.merge:
        merged = {}
        for path in args.merge:
            with open(path) as fh:
                d = json.load(fh)
            n = d["world_size"]
            key = "intra_node_GBps" if n <= 4 else "inter_node_GBps"
            merged[key] = d["peak_algbw_GBps"]
            merged[f"detail_{n}gpu"] = d
        merged.setdefault("inter_node_GBps", merged.get("intra_node_GBps"))
        with open(args.out, "w") as fh:
            json.dump(merged, fh, indent=2)
        print(json.dumps({k: v for k, v in merged.items()
                          if not k.startswith("detail")}, indent=2))
        print(f"wrote {args.out}")
        return

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size < 2:
        raise SystemExit("needs at least 2 ranks; launch it with srun -n 2 or more")

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend=backend)
    master = rank == 0

    sizes = args.sizes_mb or DEFAULT_SIZES_MB
    if master:
        print(f"world_size {world_size} | backend {backend} | device {device}")
        print(f"{'size MB':>9s} {'time ms':>9s} {'alg GB/s':>10s} {'bus GB/s':>10s}")

    results = []
    for mb in sizes:
        numel = int(mb * 1e6 / 4)
        t = bench_one(numel, device, args.iters, args.warmup)
        payload = numel * 4
        algbw = payload / t / 1e9
        busbw = algbw * 2 * (world_size - 1) / world_size
        if master:
            print(f"{mb:9.1f} {t*1000:9.3f} {algbw:10.2f} {busbw:10.2f}")
        results.append({"size_mb": mb, "time_s": t,
                        "algbw_GBps": algbw, "busbw_GBps": busbw})

    if master:
        peak = max(r["algbw_GBps"] for r in results)
        peak_bus = max(r["busbw_GBps"] for r in results)
        out = {"world_size": world_size, "backend": backend,
               "peak_algbw_GBps": peak, "peak_busbw_GBps": peak_bus,
               "results": results}
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\npeak algorithmic bandwidth {peak:.2f} GB/s")
        print(f"peak bus bandwidth         {peak_bus:.2f} GB/s")
        print(f"wrote {args.out}")
        print("\nUse this to predict communication time per step:")
        print("    comm_s = 2*(N-1)/N * gradient_bytes / (algbw * 1e9)")
        print("MobileNetV2 gradient 8.9 MB, ResNet-18 gradient 44.7 MB (fp32)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
