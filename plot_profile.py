"""
Turn profile_step.py output into the figures the bottleneck section needs.

Point it at the directory holding the prof-*/ folders. It finds every
breakdown.json and writes:

  fig_time_breakdown.png     stacked bar per GPU count: compute / comms / data
  fig_data_wait.png          share of each step spent waiting on the loader
  fig_pipeline_fix.png       before and after --gpu_resident, if both exist
  profile_summary.csv        one row per profiled configuration

    python plot_profile.py --runs_dir . --out_dir figures

The stacked bar is the figure the deck promises. The second one is the number
that decides whether the input pipeline or the all-reduce is the real limit:
if data wait is a large share at every GPU count, the curve you measured is
your data loader, not DDP.
"""

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOURS = {"compute": "#2E6F9E", "communication": "#E07A3F", "data": "#8A93A8"}
ORDER = ["compute", "communication", "data"]


def load(runs_dir):
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "**", "breakdown.json"),
                                 recursive=True)):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception as e:
            print(f"skipping {path}: {e}")
            continue
        d["_dir"] = os.path.dirname(path)
        # profile_step.py does not record the pipeline mode, so infer it from
        # the folder name: anything containing "gpures" is the fixed version.
        d["_pipeline"] = ("GPU-resident"
                          if "gpures" in d["_dir"].lower()
                          or "resident" in d["_dir"].lower()
                          else "CPU loader")
        out.append(d)
    return out


def key(d):
    return (d.get("model", "?"), d.get("world_size", 1), d["_pipeline"])


def fig_breakdown(runs, out):
    runs = sorted(runs, key=lambda d: (d.get("model", ""), d.get("world_size", 1)))
    if not runs:
        return
    labels, bottoms = [], []
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(runs) + 3), 5))
    xs = range(len(runs))
    stack = {k: [] for k in ORDER}
    for d in runs:
        frac = d.get("profiler_bucket_fraction", {})
        for k in ORDER:
            stack[k].append(frac.get(k, 0.0) * 100)
        labels.append(f"{d.get('model','?').replace('_v2','V2')}\n"
                      f"{d.get('world_size',1)} GPU\n{d['_pipeline']}")
    base = [0.0] * len(runs)
    for k in ORDER:
        ax.bar(xs, stack[k], bottom=base, label=k.capitalize(),
               color=COLOURS[k], width=0.62)
        base = [b + v for b, v in zip(base, stack[k])]
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Share of profiled step time (%)")
    ax.set_xlabel("Configuration")
    ax.set_title("Where each training step spends its time")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_data_wait(runs, out):
    groups = {}
    for d in runs:
        groups.setdefault((d.get("model", "?"), d["_pipeline"]), []).append(d)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plotted = False
    for (model, pipe), rs in sorted(groups.items()):
        rs.sort(key=lambda d: d.get("world_size", 1))
        xs = [d.get("world_size", 1) for d in rs]
        ys = [d.get("data_wait_fraction", 0) * 100 for d in rs]
        if len(xs) < 1:
            continue
        ax.plot(xs, ys, marker="o" if pipe == "CPU loader" else "s",
                linestyle="-" if pipe == "CPU loader" else "--",
                label=f"{model}, {pipe}")
        plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Share of each step waiting on data (%)")
    ax.set_title("Is the input pipeline the limit?")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_fix(runs, out):
    """Before-and-after, only if both pipeline variants were profiled."""
    pairs = {}
    for d in runs:
        k = (d.get("model", "?"), d.get("world_size", 1))
        pairs.setdefault(k, {})[d["_pipeline"]] = d
    both = {k: v for k, v in pairs.items() if len(v) == 2}
    if not both:
        print("no before/after pair yet (profile with and without --gpu_resident); "
              "pipeline-fix figure skipped")
        return
    labels, before, after = [], [], []
    for (model, ws), v in sorted(both.items()):
        labels.append(f"{model}\n{ws} GPU")
        before.append(v["CPU loader"]["mean_step_ms"])
        after.append(v["GPU-resident"]["mean_step_ms"])
    xs = range(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(6.5, 1.6 * len(labels) + 3), 5))
    ax.bar([x - w / 2 for x in xs], before, w, label="CPU loader", color="#8A93A8")
    ax.bar([x + w / 2 for x in xs], after, w, label="GPU-resident", color="#2E6F9E")
    for x, b, a in zip(xs, before, after):
        if b > 0:
            ax.text(x, max(b, a) * 1.02, f"{(b/a - 1)*100:+.0f}%",
                    ha="center", fontsize=9, color="#14213D")
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean step time (ms)")
    ax.set_xlabel("Configuration")
    ax.set_title("Effect of moving the input pipeline onto the GPU")
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def write_csv(runs, path):
    cols = ["model", "gpus", "pipeline", "per_gpu_batch", "precision",
            "workers", "bucket_mb", "mean_step_ms", "mean_data_wait_ms",
            "data_wait_pct", "compute_pct", "comms_pct", "data_pct", "dir"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for d in sorted(runs, key=key):
            f = d.get("profiler_bucket_fraction", {})
            w.writerow([
                d.get("model", ""), d.get("world_size", ""), d["_pipeline"],
                d.get("per_gpu_batch", ""), d.get("precision", ""),
                d.get("workers", ""), d.get("bucket_mb", ""),
                round(d.get("mean_step_ms", 0), 2),
                round(d.get("mean_data_wait_ms", 0), 2),
                round(d.get("data_wait_fraction", 0) * 100, 1),
                round(f.get("compute", 0) * 100, 1),
                round(f.get("communication", 0) * 100, 1),
                round(f.get("data", 0) * 100, 1),
                d["_dir"],
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=".")
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()

    runs = load(args.runs_dir)
    if not runs:
        raise SystemExit(f"no breakdown.json found under {args.runs_dir}")

    print(f"found {len(runs)} profiled configurations:")
    for d in sorted(runs, key=key):
        f = d.get("profiler_bucket_fraction", {})
        print(f"  {d.get('model','?'):<14s} {d.get('world_size',1):>2d} GPU  "
              f"{d['_pipeline']:<13s} step {d.get('mean_step_ms',0):7.2f} ms  "
              f"compute {f.get('compute',0)*100:5.1f}%  "
              f"comms {f.get('communication',0)*100:5.1f}%  "
              f"data {f.get('data',0)*100:5.1f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    o = lambda n: os.path.join(args.out_dir, n)
    fig_breakdown(runs, o("fig_time_breakdown.png"))
    fig_data_wait(runs, o("fig_data_wait.png"))
    fig_fix(runs, o("fig_pipeline_fix.png"))
    write_csv(runs, o("profile_summary.csv"))
    print(f"\nwrote figures and profile_summary.csv to {args.out_dir}/")


if __name__ == "__main__":
    main()
