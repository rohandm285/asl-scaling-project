"""
Evaluate a trained checkpoint and produce the numbers the report needs.

train_asl.py only ever looks at the validation split. This script handles
everything else:

  * accuracy on the held-out TEST split
  * accuracy on a completely separate folder (our own-hands photos)
  * per-class accuracy, so we can see which letters fail
  * a confusion matrix, saved as both .npy and .png

Three ways to run it:

  # 1. the test split from a splits dir
  python eval_asl.py --ckpt out-baseline-mobilenet_v2/best.pt --splits_dir splits_block \\
      --split test --data_root $SCRATCH/asl/asl_alphabet_train/asl_alphabet_train

  # 2. a loose folder of images, one subfolder per class (our own hands)
  python eval_asl.py --ckpt out-baseline-mobilenet_v2/best.pt --image_dir ourhands

  # 3. the block-vs-random split comparison, the headline leakage result
  python eval_asl.py --ckpt out-baseline-mobilenet_v2/best.pt --splits_dir splits_random \\
      --split test --data_root ... --tag random
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def build_model(name, num_classes):
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    elif name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "scratch_cnn":
        m = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes),
        )
    else:
        raise ValueError(name)
    return m


class EvalDataset(Dataset):
    """Either a (root, list-file) pair or a folder tree of class subfolders."""

    def __init__(self, items, transform):
        self.items = items          # list of (abs_path, label)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def items_from_list(data_root, list_file):
    out = []
    with open(list_file) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            rel, label = line.split("\t")
            out.append((os.path.join(data_root, rel), int(label)))
    return out


def items_from_dir(image_dir, class_to_idx):
    out, skipped = [], []
    for cls in sorted(os.listdir(image_dir)):
        d = os.path.join(image_dir, cls)
        if not os.path.isdir(d):
            continue
        if cls not in class_to_idx:
            skipped.append(cls)
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append((os.path.join(d, f), class_to_idx[cls]))
    if skipped:
        print(f"WARNING: folders not matching any class name, ignored: {skipped}")
        print(f"         expected names: {sorted(class_to_idx)}")
    return out


def save_confusion_png(cm, classes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Row-normalise so classes with different counts are comparable.
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = cm / np.maximum(1, cm.sum(axis=1, keepdims=True))

    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.42), max(6, n * 0.40)))
    im = ax.imshow(cmn, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits_dir", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--image_dir", default=None,
                    help="Folder with one subfolder per class; use instead of --splits_dir")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out_dir", default="eval")
    ap.add_argument("--tag", default=None, help="Suffix for the output file names")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    img_size = ckpt.get("img_size", 224)

    model = build_model(ckpt["model_name"], len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    tf = T.Compose([
        T.Resize((img_size, img_size)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.image_dir:
        items = items_from_dir(args.image_dir, class_to_idx)
        source = os.path.abspath(args.image_dir)
        tag = args.tag or os.path.basename(os.path.normpath(args.image_dir))
    else:
        if not args.splits_dir:
            raise SystemExit("pass either --splits_dir or --image_dir")
        meta = json.load(open(os.path.join(args.splits_dir, "meta.json")))
        data_root = args.data_root or meta["data_root"]
        if not os.path.isdir(data_root):
            raise SystemExit(
                f"data_root does not exist: {data_root}\n"
                f"Pass --data_root pointing at YOUR copy of asl_alphabet_train.")
        items = items_from_list(data_root, os.path.join(args.splits_dir, f"{args.split}.txt"))
        source = f"{args.splits_dir}:{args.split}"
        tag = args.tag or f"{os.path.basename(os.path.normpath(args.splits_dir))}_{args.split}"

    if not items:
        raise SystemExit(f"no images found for {source}")

    loader = DataLoader(EvalDataset(items, tf), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers,
                        pin_memory=(device.type == "cuda"))

    n_cls = len(classes)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    correct = total = 0
    conf_correct, conf_wrong = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            conf, pred = probs.max(dim=1)
            pred = pred.cpu().numpy()
            conf = conf.cpu().numpy()
            y = y.numpy()
            for t, p, c in zip(y, pred, conf):
                cm[t, p] += 1
                if t == p:
                    conf_correct.append(float(c))
                else:
                    conf_wrong.append(float(c))
            correct += int((pred == y).sum())
            total += len(y)

    acc = correct / total
    per_class = {}
    for i, c in enumerate(classes):
        n = int(cm[i].sum())
        per_class[c] = {"n": n, "acc": float(cm[i, i] / n) if n else None}

    # The most confused pairs -- the headline of the error analysis.
    pairs = []
    for i in range(n_cls):
        for j in range(n_cls):
            if i != j and cm[i, j] > 0:
                pairs.append((classes[i], classes[j], int(cm[i, j])))
    pairs.sort(key=lambda t: -t[2])

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, f"confusion_{tag}.npy"), cm)
    try:
        save_confusion_png(cm, classes,
                           os.path.join(args.out_dir, f"confusion_{tag}.png"),
                           f"Confusion matrix: {tag} (acc {acc*100:.1f}%)")
    except Exception as e:
        print(f"(confusion png skipped: {e})")

    result = {
        "checkpoint": os.path.abspath(args.ckpt),
        "model": ckpt["model_name"],
        "source": source,
        "n_images": total,
        "accuracy": acc,
        "mean_confidence_correct": float(np.mean(conf_correct)) if conf_correct else None,
        "mean_confidence_wrong": float(np.mean(conf_wrong)) if conf_wrong else None,
        "per_class": per_class,
        "top_confusions": pairs[:20],
    }
    with open(os.path.join(args.out_dir, f"eval_{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n=== {tag} ===")
    print(f"model      : {ckpt['model_name']}")
    print(f"source     : {source}")
    print(f"images     : {total}")
    print(f"ACCURACY   : {acc*100:.2f}%")
    if conf_wrong:
        print(f"mean confidence  correct {np.mean(conf_correct)*100:5.1f}%  "
              f"wrong {np.mean(conf_wrong)*100:5.1f}%")
    worst = sorted((v["acc"], k) for k, v in per_class.items() if v["acc"] is not None)[:8]
    print("\nworst classes:")
    for a, c in worst:
        print(f"  {c:>8s}  {a*100:5.1f}%")
    print("\ntop confusions (true -> predicted, count):")
    for t, pr, n in pairs[:8]:
        print(f"  {t:>8s} -> {pr:<8s} {n}")
    print(f"\nwrote {args.out_dir}/eval_{tag}.json and confusion_{tag}.(npy|png)")


if __name__ == "__main__":
    main()
