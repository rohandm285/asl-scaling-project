"""
GPU-resident input pipeline: the fix for the data-loading bottleneck.

THE PROBLEM
-----------
The normal pipeline reads 87,000 small JPEGs off Lustre, decodes them on the
CPU, and augments them there too. A small model like MobileNetV2 finishes a
step in a few milliseconds, which is faster than a handful of CPU workers can
decode a batch. The GPUs then sit idle waiting for images, and the scaling
curve you measure is really a curve of your data loader, not of DDP.

Lustre makes it worse: a parallel filesystem is built for a few large files,
not for tens of thousands of tiny ones, so the per-file overhead dominates.

THE FIX
-------
The whole training set is about 10 GB as raw 8-bit pixels at the dataset's
native 200x200, which fits in a 40 GB A100 alongside the model. So:

  1. Decode every image ONCE at startup, into a uint8 tensor on the GPU.
  2. Each rank holds only its own shard, so with 4 GPUs each stores ~2.6 GB.
  3. Every batch after that is a GPU-side index into that tensor. No file
     reads, no JPEG decoding, no host-to-device copies during training.
  4. Augmentation runs on the GPU as batched tensor ops.

The point is not that this is the right engineering choice in general -- it
only works because the dataset is small enough to fit. The point is to measure
how much of the measured step time was the input pipeline. Report throughput
with and without `--gpu_resident` and the difference is the answer.

WHAT TO WATCH
-------------
Startup costs a few minutes of decoding, so this pays off for long runs and
costs you for short ones. The loader prints the time and the memory it used.
If the shard does not fit, it says so rather than crashing with an OOM
halfway through training.
"""

import os
import time

import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _read_list(data_root, list_file):
    items = []
    with open(list_file) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            rel, label = line.split("\t")
            items.append((os.path.join(data_root, rel), int(label)))
    return items


class GpuResidentLoader:
    """
    Iterable yielding (images, labels) batches, both already on the GPU.

    Drop-in for a DataLoader in the training loop: supports len() and
    iteration, and like the DataLoader with drop_last=True it yields only full
    batches.
    """

    def __init__(self, data_root, list_file, batch_size, img_size, device,
                 rank=0, world_size=1, aug="heavy", store_size=None,
                 verbose=True, seed=0):
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device
        self.aug = aug
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.seed = seed

        items = _read_list(data_root, list_file)
        # Shard deterministically so every rank sees a disjoint slice.
        shard = items[rank::world_size]
        if not shard:
            raise RuntimeError(f"rank {rank}: empty shard from {list_file}")

        # Store at the dataset's native resolution unless told otherwise, then
        # crop and resize to img_size on the GPU. Storing at 200 rather than
        # 224 saves about 20% of the memory.
        if store_size is None:
            with Image.open(shard[0][0]) as probe:
                store_size = min(probe.size)
        self.store_size = store_size

        n = len(shard)
        need_gb = n * store_size * store_size * 3 / 1e9
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
            if need_gb * 1e9 > free * 0.80:
                raise RuntimeError(
                    f"GPU-resident pipeline needs ~{need_gb:.1f} GB for this "
                    f"rank's {n} images at {store_size}px, but only "
                    f"{free/1e9:.1f} GB is free. Use more GPUs so each shard is "
                    f"smaller, pass a smaller --img_size, or drop "
                    f"--gpu_resident.")

        if verbose:
            print(f"GPU-resident pipeline: decoding {n} images per rank at "
                  f"{store_size}px (~{need_gb:.2f} GB/rank)...", flush=True)

        t0 = time.time()
        self.images = torch.empty((n, 3, store_size, store_size),
                                  dtype=torch.uint8, device=device)
        labels = torch.empty(n, dtype=torch.long)

        # Decode in chunks so the host-side staging buffer stays small.
        CHUNK = 512
        buf = torch.empty((CHUNK, 3, store_size, store_size), dtype=torch.uint8)
        filled = 0
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            for i in range(start, end):
                path, lab = shard[i]
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    if im.size != (store_size, store_size):
                        im = im.resize((store_size, store_size), Image.BILINEAR)
                    # bytearray makes the buffer writable, which avoids a
                    # harmless-but-noisy PyTorch warning.
                    t = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
                    t = t.view(store_size, store_size, 3).permute(2, 0, 1)
                buf[i - start].copy_(t)
                labels[i] = lab
            self.images[start:end].copy_(buf[: end - start], non_blocking=True)
            filled = end
            if verbose and (start // CHUNK) % 20 == 0 and start:
                print(f"  {filled}/{n} decoded", flush=True)

        self.labels = labels.to(device)
        self.n = n
        if device.type == "cuda":
            torch.cuda.synchronize()
        self.load_seconds = time.time() - t0
        if verbose:
            print(f"GPU-resident pipeline ready in {self.load_seconds:.1f}s "
                  f"({n} images, {need_gb:.2f} GB/rank). The CPU input pipeline "
                  f"is now out of the loop.", flush=True)

        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __len__(self):
        return self.n // self.batch_size

    def set_epoch(self, epoch):
        """Matches DistributedSampler's interface so the training loop is unchanged."""
        self.epoch = epoch

    # ---------------- GPU-side augmentation ----------------

    def _augment(self, batch_u8):
        """
        batch_u8: (B, 3, S, S) uint8 on the GPU. Returns normalized float.

        Mirrors the CPU transforms closely enough for a fair comparison:
        random resized crop, rotation, colour jitter, random grayscale.
        No horizontal flip -- mirroring a hand changes its handedness, which
        would teach the model wrong shapes.
        """
        B = batch_u8.shape[0]
        S = self.store_size
        out_size = self.img_size
        dev = self.device
        x = batch_u8.float().div_(255.0)

        if self.aug == "none":
            if S != out_size:
                x = F.interpolate(x, size=(out_size, out_size),
                                  mode="bilinear", align_corners=False)
            return (x - self.mean) / self.std

        heavy = self.aug == "heavy"
        scale_lo = 0.7 if heavy else 0.85
        max_deg = 15.0 if heavy else 0.0
        jitter = 0.3 if heavy else 0.2

        # --- random resized crop + rotation, as one affine grid_sample ---
        scale = torch.empty(B, device=dev).uniform_(scale_lo, 1.0).sqrt()
        tx = (1 - scale) * torch.empty(B, device=dev).uniform_(-1, 1)
        ty = (1 - scale) * torch.empty(B, device=dev).uniform_(-1, 1)
        if max_deg > 0:
            ang = torch.empty(B, device=dev).uniform_(-max_deg, max_deg) * torch.pi / 180
        else:
            ang = torch.zeros(B, device=dev)
        cos, sin = torch.cos(ang), torch.sin(ang)

        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = cos * scale
        theta[:, 0, 1] = -sin * scale
        theta[:, 1, 0] = sin * scale
        theta[:, 1, 1] = cos * scale
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta, (B, 3, out_size, out_size), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                          align_corners=False)

        # --- colour jitter: brightness, contrast, saturation ---
        b = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - jitter, 1 + jitter)
        x = x * b
        c = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - jitter, 1 + jitter)
        mean_pix = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean_pix) * c + mean_pix
        gray = (x * torch.tensor([0.299, 0.587, 0.114], device=dev)
                .view(1, 3, 1, 1)).sum(1, keepdim=True)
        s = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - jitter, 1 + jitter)
        x = gray + (x - gray) * s

        # --- random grayscale on a fraction of the batch ---
        if heavy:
            g_mask = (torch.rand(B, 1, 1, 1, device=dev) < 0.2).float()
            x = x * (1 - g_mask) + gray.expand_as(x) * g_mask

        x = x.clamp_(0, 1)
        return (x - self.mean) / self.std

    def __iter__(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch * 1000 + self.rank)
        perm = torch.randperm(self.n, generator=g).to(self.device)
        nb = len(self)
        for i in range(nb):
            idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
            yield self._augment(self.images[idx]), self.labels[idx]
        self.epoch += 1
