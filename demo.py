"""
Live webcam demo. Runs on your laptop CPU from the checkpoint trained on Perlmutter.

    pip install torch torchvision opencv-python mediapipe
    python demo.py --ckpt best.pt

Press q to quit, SPACE to clear the spelled text.

Hand cropping: uses MediaPipe if it is installed, otherwise falls back to a
fixed box in the middle of the frame that you hold your hand inside. The
fallback works fine for a demo -- just tell the audience to aim for the box.

Prediction smoothing: a single frame's guess flickers badly. We keep a short
history and only commit a letter once the same class has won a majority of the
window AND the confidence is above a threshold, then require the hand to leave
or change before the next letter can be committed. That is the cheap version of
the temporal segmentation stage.
"""

import argparse
import collections
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


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


class HandCropper:
    """MediaPipe hand box if available, else a fixed centre box."""

    def __init__(self, margin=0.35):
        self.margin = margin
        self.hands = None
        try:
            import mediapipe as mp
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
            print("hand detection: mediapipe")
        except Exception as e:
            print(f"hand detection: fixed box (mediapipe unavailable: {e})")

    def __call__(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if self.hands is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.hands.process(rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                xs = [p.x for p in lm]
                ys = [p.y for p in lm]
                x0, x1 = min(xs) * w, max(xs) * w
                y0, y1 = min(ys) * h, max(ys) * h
                bw, bh = x1 - x0, y1 - y0
                side = max(bw, bh) * (1 + self.margin)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                x0 = int(max(0, cx - side / 2))
                y0 = int(max(0, cy - side / 2))
                x1 = int(min(w, cx + side / 2))
                y1 = int(min(h, cy + side / 2))
                if x1 - x0 > 20 and y1 - y0 > 20:
                    return frame_bgr[y0:y1, x0:x1], (x0, y0, x1, y1), True

        side = int(min(h, w) * 0.5)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        return frame_bgr[y0:y0 + side, x0:x0 + side], (x0, y0, x0 + side, y0 + side), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.70,
                    help="confidence needed before a letter is committed")
    ap.add_argument("--window", type=int, default=12,
                    help="frames of history used for the majority vote")
    ap.add_argument("--mirror", action="store_true", default=True)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 224)
    model = build_model(ckpt["model_name"], len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"loaded {ckpt['model_name']} | {len(classes)} classes | "
          f"val_acc {ckpt.get('val_acc', float('nan')):.4f}")

    tf = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cropper = HandCropper()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    history = collections.deque(maxlen=args.window)
    spelled = ""
    last_committed = None
    fps_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        crop, (x0, y0, x1, y1), detected = cropper(frame)

        label, conf = "-", 0.0
        if crop.size > 0:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                logits = model(tf(rgb).unsqueeze(0))
                probs = torch.softmax(logits, dim=1)[0]
                idx = int(probs.argmax())
                label, conf = classes[idx], float(probs[idx])
            history.append((label, conf))

        # ---- commit logic: majority vote over the window, above threshold ----
        if len(history) == history.maxlen:
            votes = collections.Counter(l for l, c in history if c >= args.conf)
            if votes:
                top, count = votes.most_common(1)[0]
                if count >= history.maxlen * 0.7 and top != last_committed:
                    if top == "space":
                        spelled += " "
                    elif top == "del":
                        spelled = spelled[:-1]
                    elif top != "nothing":
                        spelled += top
                    last_committed = top
                    history.clear()
            # seeing "nothing" clears the lock so the same letter can repeat
            if any(l == "nothing" for l, _ in history):
                last_committed = None

        # ---- drawing ----
        colour = (0, 220, 0) if detected else (0, 165, 255)
        cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (25, 25, 35), -1)
        cv2.putText(frame, f"{label}  {conf*100:4.1f}%", (14, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (240, 240, 240), 2)
        cv2.putText(frame, f"{fps:4.1f} fps", (frame.shape[1] - 150, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 160), 2)

        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 70), (frame.shape[1], h), (25, 25, 35), -1)
        cv2.putText(frame, spelled[-28:], (14, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (120, 220, 255), 2)

        cv2.imshow("ASL fingerspelling", frame)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - fps_t))
        fps_t = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            spelled = ""
            last_committed = None
            history.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
