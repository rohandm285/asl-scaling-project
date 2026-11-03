"""
Capture our own held-out test set from a laptop webcam.

This is the data behind the strongest finding in the project: how much of the
dataset accuracy survives on a hand the model has never seen, in a room it has
never seen. It needs all four of us, and it takes about twenty minutes.

    pip install opencv-python mediapipe
    python capture_hands.py --person vincent --out ourhands

It walks through all 29 classes. For each one it shows you the letter, waits
for you to get your hand into position, then captures a burst of frames a few
hundred milliseconds apart so you get natural variation rather than 3 copies
of one pose.

    SPACE   start the burst for the current letter
    s       skip this letter
    b       go back one letter
    q       quit (progress is kept)

Output layout matches the dataset, so eval_asl.py reads it directly:

    ourhands/
      A/vincent_A_000.jpg ...
      B/...

TIPS FOR A USEFUL TEST SET
--------------------------
Vary it deliberately. Move around the room between letters, change the
distance to the camera, let the lighting differ between people. A test set
captured as 29 identical well-lit poses tells you almost nothing; the point is
to be harder than the training data, not easier.

J and Z involve motion in real ASL, so a still frame cannot capture them
properly. Capture them anyway at the pose the dataset uses, and exclude them
when you report demo accuracy.
"""

import argparse
import os
import time

import cv2

CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + ["space", "del", "nothing"]
MOTION_LETTERS = {"J", "Z"}


def draw_banner(frame, lines, origin=(14, 40), scale=0.8, colour=(240, 240, 240)):
    y = origin[1]
    for text, sc, col in lines:
        cv2.putText(frame, text, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX,
                    sc, col, 2, cv2.LINE_AA)
        y += int(38 * sc / 0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True,
                    help="Your name, used in the file names so we can tell whose hand is whose")
    ap.add_argument("--out", default="ourhands")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--shots", type=int, default=3, help="Frames per letter")
    ap.add_argument("--burst_gap", type=float, default=0.6,
                    help="Seconds between frames in a burst")
    ap.add_argument("--size", type=int, default=400,
                    help="Saved crop size in pixels (square)")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="Only capture these classes, e.g. --classes A B C")
    args = ap.parse_args()

    classes = args.classes or CLASSES
    os.makedirs(args.out, exist_ok=True)
    for c in classes:
        os.makedirs(os.path.join(args.out, c), exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    idx = 0
    captured = {c: 0 for c in classes}
    # Count anything already on disk so a second session adds to the first.
    for c in classes:
        d = os.path.join(args.out, c)
        captured[c] = len([f for f in os.listdir(d)
                           if f.startswith(args.person + "_")])

    print(f"capturing for '{args.person}' into {args.out}/")
    print("SPACE capture  ·  s skip  ·  b back  ·  q quit\n")

    bursting = False
    burst_left = 0
    next_shot = 0.0

    while idx < len(classes):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        cls = classes[idx]

        # Centre box: what gets saved.
        side = min(h, w) - 80
        x0, y0 = (w - side) // 2, (h - side) // 2
        box_col = (0, 200, 255) if bursting else (0, 220, 0)
        cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), box_col, 3)

        note = "  (motion letter - pose only)" if cls in MOTION_LETTERS else ""
        draw_banner(frame, [
            (f"{cls}{note}", 1.5, (255, 255, 255)),
            (f"{idx+1} of {len(classes)}   saved so far: {captured[cls]}", 0.7,
             (170, 180, 200)),
            ("SPACE capture   s skip   b back   q quit", 0.6, (140, 150, 170)),
        ])

        if bursting:
            cv2.putText(frame, f"capturing {args.shots - burst_left + 1}/{args.shots}",
                        (x0 + 10, y0 + side - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 200, 255), 2)

        cv2.imshow("capture own-hands set", frame)
        key = cv2.waitKey(1) & 0xFF

        if bursting:
            if time.time() >= next_shot:
                crop = frame[y0:y0 + side, x0:x0 + side]
                # Re-crop from a clean frame so the overlay is not baked in.
                ok2, clean = cap.read()
                if ok2:
                    clean = cv2.flip(clean, 1)
                    crop = clean[y0:y0 + side, x0:x0 + side]
                crop = cv2.resize(crop, (args.size, args.size),
                                  interpolation=cv2.INTER_AREA)
                n = captured[cls]
                path = os.path.join(args.out, cls,
                                    f"{args.person}_{cls}_{n:03d}.jpg")
                cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                captured[cls] = n + 1
                burst_left -= 1
                next_shot = time.time() + args.burst_gap
                if burst_left <= 0:
                    bursting = False
                    idx += 1
            continue

        if key == ord(" "):
            bursting = True
            burst_left = args.shots
            next_shot = time.time() + 0.35
        elif key == ord("s"):
            idx += 1
        elif key == ord("b"):
            idx = max(0, idx - 1)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    total = sum(captured.values())
    missing = [c for c in classes if captured[c] == 0]
    print(f"\nsaved {total} images for '{args.person}' in {args.out}/")
    if missing:
        print(f"no images yet for: {', '.join(missing)}")
    print("\nWhen everyone has captured their set, evaluate with:")
    print(f"    python eval_asl.py --ckpt out-.../best.pt --image_dir {args.out}")


if __name__ == "__main__":
    main()
