"""
Build train / val / test file lists for the ASL Alphabet dataset.

Why this script exists instead of a plain random split:

The ASL Alphabet images are consecutive frames pulled from short videos.
Within one class folder, A1.jpg and A2.jpg are nearly the same picture. If you
shuffle all 3000 images in a class and split randomly, almost every test image
has a near-twin sitting in the training set, and the model scores ~99.9% by
recognising frames it has effectively already seen. That number is meaningless.

So we split by CONTIGUOUS BLOCKS of the frame index instead. Train gets the
first chunk of each class, val the next, test the last. Frames that are
adjacent in time stay on the same side of the split, which removes most of the
leakage. We report both splits so you can show the difference in the report.

Usage:
    python prepare_data.py --data_root $SCRATCH/asl/asl_alphabet_train/asl_alphabet_train \
                           --out_dir splits_block --seed 0
    python prepare_data.py --data_root $SCRATCH/asl/asl_alphabet_train/asl_alphabet_train \
                           --out_dir splits_random --mode random --seed 0
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict


def natural_key(name):
    """Sort A1.jpg, A2.jpg, ..., A10.jpg in frame order, not lexicographic."""
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="Folder holding one subfolder per class (A, B, ... nothing)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_frac", type=float, default=0.80)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--mode", choices=["block", "random"], default="block",
                    help="block = leak-aware contiguous split; random = naive shuffle")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    classes = sorted(
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
    )
    if not classes:
        raise SystemExit(f"No class folders found under {args.data_root}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    splits = defaultdict(list)
    per_class_counts = {}

    for cls in classes:
        cls_dir = os.path.join(args.data_root, cls)
        files = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        files.sort(key=natural_key)

        if args.mode == "random":
            random.shuffle(files)

        n = len(files)
        n_train = int(n * args.train_frac)
        n_val = int(n * args.val_frac)

        chunks = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        per_class_counts[cls] = {k: len(v) for k, v in chunks.items()}

        for split, names in chunks.items():
            for f in names:
                rel = os.path.join(cls, f)
                splits[split].append(f"{rel}\t{class_to_idx[cls]}")

    for split, rows in splits.items():
        # Shuffle the written order so the DataLoader sees mixed classes even
        # without shuffling; harmless and makes debugging output readable.
        random.shuffle(rows)
        path = os.path.join(args.out_dir, f"{split}.txt")
        with open(path, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        print(f"{split:5s}: {len(rows):7d} images -> {path}")

    meta = {
        "data_root": os.path.abspath(args.data_root),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "num_classes": len(classes),
        "split_mode": args.mode,
        "per_class_counts": per_class_counts,
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n{len(classes)} classes -> {meta_path}")
    print(f"split mode: {args.mode}")
    if args.mode == "block":
        print("Contiguous blocks: adjacent video frames stay on one side of the split.")
    else:
        print("WARNING: random split leaks near-duplicate frames. Accuracy will be inflated.")


if __name__ == "__main__":
    main()
