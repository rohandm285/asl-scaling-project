"""
Turn a pile of history.json files into the figures the report needs.

Point it at the directory holding all the out-*/ run folders. It finds every
history.json underneath, groups runs by model and by scaling mode, and writes:

  fig_throughput.png        images/sec vs GPU count
  fig_speedup.png           speedup vs GPU count, with the ideal line
  fig_efficiency.png        parallel efficiency (speedup / GPUs) vs GPU count
  fig_time_to_acc.png       wall-clock seconds to the target accuracy
  fig_acc_vs_batch.png      best validation accuracy vs global batch size
  fig_loss_curves.png       validation accuracy against epoch, per run
  summary.csv               every run as one row, for the report tables

Optionally pass --bandwidth_json from nccl_bench.py and it adds:

  fig_predicted_vs_measured.png

which is the plot that makes the scaling section a prediction rather than a
measurement.

    python plot_results.py --runs_dir . --out_dir figures \\
        --bandwidth_json nccl_results.json
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One colour per model, consistent across every figure.
COLOURS = {"mobilenet_v2": "#1f77b4", "resnet18": "#ff7f0e",
           "resnet50": "#2ca02c", "scratch_cnn": "#9467bd"}
MARKERS = {"strong": "o", "weak": "s", "unknown": "^"}


def load_runs(runs_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "**", "history.json"),
                                 recursive=True)):
        try:
            with open(path) as fh:
                h = json.load(fh)
        except Exception as e:
            print(f"skipping unreadable {path}: {e}")
            continue
        a = h.get("args", {})
        # Weak scaling keeps per-GPU batch fixed, so global batch grows with
        # world size. Strong scaling keeps global batch fixed.
        ws = h.get("world_size", 1)
        gb = h.get("global_batch", a.get("batch_size", 0) * ws)
        # Prefer an explicit --tag; otherwise infer. A timed sweep that holds
        # per-GPU batch constant is weak scaling even without --scale_lr.
        h["_mode"] = (a.get("tag")
                      or ("weak" if a.get("scale_lr") else "strong"))
        h["_timed"] = h.get("mode") == "timed"
        h["_model"] = a.get("model", "unknown")
        h["_ws"] = ws
        h["_gb"] = gb
        h["_dir"] = os.path.dirname(path)
        runs.append(h)
    return runs


def group(runs, key):
    g = defaultdict(list)
    for r in runs:
        g[key(r)].append(r)
    for k in g:
        g[k].sort(key=lambda r: r["_ws"])
    return g


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def fig_throughput(runs, out):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (model, mode), rs in sorted(group(runs, lambda r: (r["_model"], r["_mode"])).items()):
        xs = [r["_ws"] for r in rs]
        ys = [r.get("median_img_s", 0) for r in rs]
        ax.plot(xs, ys, marker=MARKERS.get(mode, "^"), linestyle="-",
                color=COLOURS.get(model, "#555"),
                label=f"{model}, {mode} scaling")
    ax.set_xscale("log", base=2)
    style(ax, "Number of GPUs", "Throughput (training images per second)",
          "Training throughput vs GPU count")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_speedup_and_efficiency(runs, out_speedup, out_eff):
    f1, a1 = plt.subplots(figsize=(7.5, 5))
    f2, a2 = plt.subplots(figsize=(7.5, 5))
    max_ws = max([r["_ws"] for r in runs] + [1])

    for (model, mode), rs in sorted(group(runs, lambda r: (r["_model"], r["_mode"])).items()):
        base = next((r for r in rs if r["_ws"] == 1), None)
        if base is None or not base.get("median_img_s"):
            print(f"no 1-GPU baseline for {model}/{mode}; speedup skipped")
            continue
        b = base["median_img_s"]
        xs = [r["_ws"] for r in rs]
        sp = [r.get("median_img_s", 0) / b for r in rs]
        eff = [s / w * 100 for s, w in zip(sp, xs)]
        lab = f"{model}, {mode} scaling"
        c = COLOURS.get(model, "#555")
        mk = MARKERS.get(mode, "^")
        a1.plot(xs, sp, marker=mk, color=c, label=lab)
        a2.plot(xs, eff, marker=mk, color=c, label=lab)

    ideal = [2 ** i for i in range(0, max(1, int(max_ws).bit_length()))]
    a1.plot(ideal, ideal, "k--", alpha=0.45, label="ideal (linear)")
    a2.axhline(100, color="k", linestyle="--", alpha=0.45, label="ideal (100%)")

    a1.set_xscale("log", base=2); a1.set_yscale("log", base=2)
    style(a1, "Number of GPUs", "Speedup vs 1 GPU", "Scaling speedup")
    a2.set_xscale("log", base=2); a2.set_ylim(0, 115)
    style(a2, "Number of GPUs", "Parallel efficiency (%)",
          "Parallel efficiency = speedup / GPU count")

    f1.tight_layout(); f1.savefig(out_speedup, dpi=200); plt.close(f1)
    f2.tight_layout(); f2.savefig(out_eff, dpi=200); plt.close(f2)


def fig_time_to_acc(runs, out):
    have = [r for r in runs if (r.get("time_to_target_s") or -1) > 0]
    if not have:
        print("no run recorded time_to_target_s (pass --target_acc); skipping")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (model, mode), rs in sorted(group(have, lambda r: (r["_model"], r["_mode"])).items()):
        ax.plot([r["_ws"] for r in rs], [r["time_to_target_s"] for r in rs],
                marker=MARKERS.get(mode, "^"), color=COLOURS.get(model, "#555"),
                label=f"{model}, {mode} scaling")
    ax.set_xscale("log", base=2)
    style(ax, "Number of GPUs", "Wall-clock time to target accuracy (s)",
          "Time to reach the target validation accuracy")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_acc_vs_batch(runs, out):
    # Timed sweeps never evaluate, so they have no accuracy to plot.
    runs = [r for r in runs if not r.get("_timed")]
    if not runs:
        print("no convergence runs yet; accuracy-vs-batch skipped")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (model, mode), rs in sorted(group(runs, lambda r: (r["_model"], r["_mode"])).items()):
        pts = sorted({(r["_gb"], r.get("best_val_acc", 0)) for r in rs})
        if len(pts) < 2:
            continue
        ax.plot([p[0] for p in pts], [p[1] * 100 for p in pts],
                marker=MARKERS.get(mode, "^"), color=COLOURS.get(model, "#555"),
                label=f"{model}, {'lr scaled' if mode == 'weak' else 'lr fixed'}")
    ax.set_xscale("log", base=2)
    style(ax, "Global batch size (images per optimizer step)",
          "Best validation accuracy (%)",
          "Does training faster cost accuracy?")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_curves(runs, out):
    runs = [r for r in runs if not r.get("_timed") and r.get("history")]
    if not runs:
        print("no convergence runs yet; accuracy curves skipped")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in sorted(runs, key=lambda r: (r["_model"], r["_ws"])):
        h = r.get("history", [])
        if not h:
            continue
        ax.plot([e["epoch"] for e in h], [e["val_acc"] * 100 for e in h],
                marker=".", alpha=0.85,
                label=f"{r['_model']} {r['_ws']}gpu gb{r['_gb']}")
    style(ax, "Epoch", "Validation accuracy (%)", "Validation accuracy per epoch")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_predicted(runs, bw, out):
    """
    Predict where communication overtakes compute, then plot against measured.

    Ring all-reduce moves 2*(N-1)/N * S bytes per GPU, where S is the gradient
    size. Compute time per step is taken from the measured 1-GPU throughput,
    scaled down by the per-GPU batch at each configuration.
    """
    intra = bw.get("intra_node_GBps")
    inter = bw.get("inter_node_GBps", intra)
    if not intra:
        print("bandwidth json has no intra_node_GBps; skipping prediction plot")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    plotted = False
    for (model, mode), rs in sorted(group(runs, lambda r: (r["_model"], r["_mode"])).items()):
        base = next((r for r in rs if r["_ws"] == 1), None)
        if base is None or not base.get("median_img_s"):
            continue
        grad_bytes = base.get("grad_bytes_fp32") or base.get("n_params", 0) * 4
        if not grad_bytes:
            continue
        b_ips = base["median_img_s"]

        xs, pred, meas = [], [], []
        for r in rs:
            n = r["_ws"]
            per_gpu_imgs = r["_gb"] / n
            compute_s = per_gpu_imgs / b_ips                   # seconds of compute
            gbps = intra if n <= 4 else inter                  # bytes/s
            comm_s = (2 * (n - 1) / n) * grad_bytes / (gbps * 1e9)
            # DDP overlaps comms with the backward pass, so the step is bounded
            # below by compute and above by compute + comms.
            step_s = max(compute_s, compute_s * 0.5 + comm_s)
            xs.append(n)
            pred.append(r["_gb"] / step_s)
            meas.append(r.get("median_img_s", 0))
        c = COLOURS.get(model, "#555")
        ax.plot(xs, pred, "--", color=c, alpha=0.75, marker="x",
                label=f"{model} predicted")
        ax.plot(xs, meas, "-", color=c, marker="o", label=f"{model} measured")
        plotted = True

    if not plotted:
        print("not enough data for the prediction plot")
        plt.close(fig)
        return

    ax.set_xscale("log", base=2)
    style(ax, "Number of GPUs", "Throughput (training images per second)",
          "Predicted vs measured scaling")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def write_csv(runs, path):
    cols = ["model", "mode", "gpus", "per_gpu_batch", "global_batch", "lr",
            "median_img_s", "speedup", "efficiency_pct", "best_val_acc",
            "time_to_target_s", "total_time_s", "epochs", "dir"]
    base = {}
    for r in runs:
        if r["_ws"] == 1 and r.get("median_img_s"):
            base.setdefault((r["_model"], r["_mode"]), r["median_img_s"])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in sorted(runs, key=lambda r: (r["_model"], r["_mode"], r["_ws"], r["_gb"])):
            a = r.get("args", {})
            b = base.get((r["_model"], r["_mode"]))
            ips = r.get("median_img_s", 0)
            sp = ips / b if b else ""
            eff = (sp / r["_ws"] * 100) if b else ""
            w.writerow([
                r["_model"], r["_mode"], r["_ws"], a.get("batch_size", ""),
                r["_gb"], a.get("lr", ""), round(ips, 1),
                round(sp, 3) if b else "", round(eff, 1) if b else "",
                round(r.get("best_val_acc", 0), 4),
                round(r.get("time_to_target_s") or -1, 1),
                round(r.get("total_time_s", 0), 1),
                a.get("epochs", ""), r["_dir"],
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default=".",
                    help="Directory containing the out-*/ run folders")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--bandwidth_json", default=None,
                    help="Output of nccl_bench.py, enables the prediction plot")
    args = ap.parse_args()

    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"no history.json found under {args.runs_dir}")
    print(f"found {len(runs)} runs:")
    for r in sorted(runs, key=lambda r: (r["_model"], r["_mode"], r["_ws"])):
        print(f"  {r['_model']:<14s} {r['_mode']:<6s} {r['_ws']:>2d} GPU  "
              f"gb={r['_gb']:<5d} {r.get('median_img_s', 0):8.1f} img/s  "
              f"acc={r.get('best_val_acc', 0)*100:5.2f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    o = lambda n: os.path.join(args.out_dir, n)

    fig_throughput(runs, o("fig_throughput.png"))
    fig_speedup_and_efficiency(runs, o("fig_speedup.png"), o("fig_efficiency.png"))
    fig_time_to_acc(runs, o("fig_time_to_acc.png"))
    fig_acc_vs_batch(runs, o("fig_acc_vs_batch.png"))
    fig_curves(runs, o("fig_loss_curves.png"))
    write_csv(runs, o("summary.csv"))

    if args.bandwidth_json and os.path.exists(args.bandwidth_json):
        with open(args.bandwidth_json) as fh:
            fig_predicted(runs, json.load(fh), o("fig_predicted_vs_measured.png"))

    print(f"\nwrote figures and summary.csv to {args.out_dir}/")


if __name__ == "__main__":
    main()
