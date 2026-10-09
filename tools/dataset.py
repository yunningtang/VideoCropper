"""Build a YOLO dataset of Zoom video tiles from the recordings this tool has analysed.

    python tools/dataset.py [--out data/tiles] [--step 1.0] [--synthetic 1.0]

Nobody has to draw boxes. Three sources:

* The pipeline's own answers. For every person in every analysed video, the timeline
  gives one box per moment. Frames where it was sure (plain segments) and frames it
  bridged (the window was there before and after, so it is there now, even though
  the single-frame finder lost it) both become labels. The bridged ones matter most:
  they are exactly what the finder cannot do today.
* The hand-checked boxes in tests/truth.json: the validation set, so the score means
  something.
* Synthetic frames: real frames with extra tiles pasted in, under the things that break
  edges in real recordings (a context menu, the mouse, a title bar, a strip of two tiles
  with a thin separator), and random 16:9 slide content that is NOT a tile.

Training frames within 2 s of a validation moment are dropped, so the two sets do not
share near-identical pictures.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import regress  # noqa: E402  (cached analysis per code version)
from zoomcrop import timeline  # noqa: E402
from zoomcrop.detect import iou  # noqa: E402

TRUTH = json.load(open(ROOT / "tests" / "truth.json"))
OK_FLAGS = {"bridged", "inferred", "clipped", "look", "camera_off"}   # segment flags that still mean "the box is right"
VAL_GUARD = 2.0   # s around a truth moment kept out of the training set
MIN_VISIBLE = 0.3  # a tile mostly off-screen is not labelled (too little to learn from)


def frames_at(path: Path, times: list[float]):
    """Yield (t, bgr) for the wanted times, decoding the video once."""
    want = sorted(set(round(t, 3) for t in times))
    i = 0
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        for fr in c.decode(vs):
            if i >= len(want):
                break
            t = fr.time if fr.time is not None else 0.0
            if t + 1e-3 >= want[i]:
                bgr = fr.to_ndarray(format="bgr24")
                while i < len(want) and t + 1e-3 >= want[i]:
                    yield want[i], bgr
                    i += 1


def labels_from_timeline(res: dict) -> dict[float, list[list[int]]]:
    """Per sample time: the boxes of every identified person, from their timelines."""
    W, H = res["video"]["width"], res["video"]["height"]
    people = res["participants"]
    per_pid = {}
    for p in people:
        others = {q["id"] for q in people if q["id"] != p["id"]}
        per_pid[p["id"]] = [s for s in timeline.build(res, p["id"], None, others)
                            if s["status"] == "ok" and set(s["flags"]) <= OK_FLAGS]
    out: dict[float, list[list[int]]] = {}
    for s in res["samples"]:
        t = s["t"]
        boxes = []
        for pid, segs in per_pid.items():
            b = regress.box_at(segs, t)
            if b is None:
                continue
            x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])
            if (x1 - x0) * (y1 - y0) < MIN_VISIBLE * (b[2] - b[0]) * (b[3] - b[1]):
                continue
            boxes.append([x0, y0, x1, y1])
        # two people on the same spot is a mislabel; an unidentified live window that is
        # not among the labels would teach the model to miss tiles: skip such frames
        if any(iou(a, b) > 0.2 for i, a in enumerate(boxes) for b in boxes[i + 1:]):
            continue
        stray = [tl for tl in s["tiles"] if tl["pid"] is None and tl.get("live")
                 and not any(iou(tl["box"], b) >= 0.3 for b in boxes)]
        if stray:
            continue
        out[t] = boxes
    return out


def write_sample(img_dir: Path, lab_dir: Path, name: str, bgr: np.ndarray, boxes: list[list[int]]):
    H, W = bgr.shape[:2]
    cv2.imwrite(str(img_dir / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(lab_dir / f"{name}.txt", "w") as f:
        for x0, y0, x1, y1 in boxes:
            f.write(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n")


# -- synthetic ---------------------------------------------------------------------
def _menu(w: int, h: int) -> np.ndarray:
    """A Zoom-style context menu: dark panel with light text lines and one highlighted row."""
    m = np.full((h, w, 3), random.randint(28, 48), np.uint8)
    y, row = 8, random.randint(16, 22)
    hi = random.randint(0, max(0, h // row - 1))
    i = 0
    while y + row <= h:
        if i == hi:
            m[y:y + row] = (random.randint(150, 230), random.randint(90, 140), random.randint(30, 70))
        tw = random.randint(w // 4, w - 20)
        m[y + row // 2 - 1:y + row // 2 + 1, 10:10 + tw] = random.randint(180, 230)
        if random.random() < 0.2:
            m[y + row:y + row + 1] = 80
            y += 3
        y += row
        i += 1
    return m


def rint(a: int, b: int) -> int:
    """randint that tolerates a reversed range (a narrow tile near the frame edge)."""
    return random.randint(min(a, b), max(a, b))


def _cursor(bgr: np.ndarray, x: int, y: int):
    pts = np.array([[x, y], [x, y + 17], [x + 4, y + 13], [x + 7, y + 20], [x + 10, y + 18], [x + 7, y + 12], [x + 12, y + 12]], np.int32)
    cv2.fillPoly(bgr, [pts], (255, 255, 255))
    cv2.polylines(bgr, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)


def _paste(bgr: np.ndarray, crop: np.ndarray, x0: int, y0: int, w: int, h: int, border: int):
    H, W = bgr.shape[:2]
    tile = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA if crop.shape[1] > w else cv2.INTER_LINEAR)
    if border:
        c = random.choice([0, 20, 40, 230, 255])
        cv2.rectangle(bgr, (x0 - border, y0 - border), (x0 + w + border - 1, y0 + h + border - 1), (c, c, c), border)
    vx0, vy0, vx1, vy1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if vx1 > vx0 and vy1 > vy0:
        bgr[vy0:vy1, vx0:vx1] = tile[vy0 - y0:vy1 - y0, vx0 - x0:vx1 - x0]
    return [vx0, vy0, vx1, vy1]


def synthesize(bgr: np.ndarray, boxes: list[list[int]], bank: list[np.ndarray], slides: list[np.ndarray]):
    """One synthetic frame from a real one: extra tiles pasted where nothing is, then the
    usual troublemakers drawn on top. Returns (image, boxes)."""
    bgr = bgr.copy()
    H, W = bgr.shape[:2]
    out = [list(b) for b in boxes]

    def free(b, pad=6):
        return all(iou(b, q) == 0 and (b[2] + pad <= q[0] or q[2] + pad <= b[0] or b[3] + pad <= q[1] or q[3] + pad <= b[1])
                   for q in out)

    # a strip of 1-3 tiles, sometimes partly off-screen, sometimes with a title bar
    n = random.choice([1, 1, 2, 2, 3])
    w = int(W * random.uniform(0.12, 0.38))
    h = round(w * 9 / 16)
    for _ in range(20):
        vertical = random.random() < 0.4
        x0 = rint(-w // 3, W - w * 2 // 3) if random.random() < 0.15 else rint(0, W - (w if vertical else n * (w + 2)))
        y0 = rint(0, H - (n * (h + 2) if vertical else h))
        cells = [(x0, y0 + i * (h + 2)) if vertical else (x0 + i * (w + 2), y0) for i in range(n)]
        if all(free([cx, cy, cx + w, cy + h]) for cx, cy in cells):
            break
    else:
        return None
    border = random.choice([0, 1, 1, 2])
    if random.random() < 0.3:  # floating window's title bar
        c = random.randint(20, 50)
        cv2.rectangle(bgr, (cells[0][0] - 2, cells[0][1] - 26), (cells[-1][0] + w + 2, cells[0][1]), (c, c, c), -1)
    for cx, cy in cells:
        out.append(_paste(bgr, random.choice(bank), cx, cy, w, h, border))
    # 16:9 slide artwork that is NOT a tile (a picture box, a bordered figure)
    if slides and random.random() < 0.5:
        sw = int(W * random.uniform(0.1, 0.3))
        sh = round(sw * 9 / 16)
        for _ in range(10):
            sx, sy = random.randint(0, W - sw), random.randint(0, H - sh)
            if free([sx, sy, sx + sw, sy + sh]):
                _paste(bgr, random.choice(slides), sx, sy, sw, sh, random.choice([0, 1]))
                break
    # the troublemakers, over a real or pasted tile
    target = random.choice(out)
    tx0, ty0, tx1, ty1 = target
    r = random.random()
    if r < 0.45:  # context menu covering part of the tile
        mw, mh = random.randint(140, 260), random.randint(160, 420)
        mx = rint(tx0 + (tx1 - tx0) // 4, tx1 - 10)
        my = rint(ty0 - 10, ty0 + (ty1 - ty0) // 2)
        menu = _menu(mw, mh)
        vx0, vy0, vx1, vy1 = max(0, mx), max(0, my), min(W, mx + mw), min(H, my + mh)
        if vx1 > vx0 and vy1 > vy0:
            bgr[vy0:vy1, vx0:vx1] = menu[vy0 - my:vy1 - my, vx0 - mx:vx1 - mx]
    elif r < 0.7:  # hover buttons ("Mute", "...") in the tile's top-right corner
        bw, bh = random.randint(40, 70), random.randint(16, 24)
        bx, by = tx1 - bw - 6, ty0 + 4
        if bx > tx0 and by + bh < ty1:
            cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), (200, 120, 30), -1)
            cv2.rectangle(bgr, (bx - 26, by), (bx - 6, by + bh), (200, 120, 30), -1)
    if random.random() < 0.5:
        _cursor(bgr, rint(max(0, tx0 - 20), min(W - 14, tx1 + 20)), rint(max(0, ty0 - 20), min(H - 22, ty1 + 20)))
    return bgr, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "tiles"))
    ap.add_argument("--step", type=float, default=1.0, help="seconds between training frames")
    ap.add_argument("--synthetic", type=float, default=1.0, help="synthetic frames per real training frame")
    ap.add_argument("--videos", nargs="*", default=list(TRUTH["videos"]))
    ap.add_argument("--synthetic-only", action="store_true", help="redo the synthetic frames from the real ones already on disk")
    args = ap.parse_args()
    random.seed(0)
    out = Path(args.out)
    dirs = {}
    for split in ("train", "val"):
        dirs[split] = (out / "images" / split, out / "labels" / split)
        for d in dirs[split]:
            d.mkdir(parents=True, exist_ok=True)

    bank: list[np.ndarray] = []    # tile pictures to paste
    slides: list[np.ndarray] = []  # 16:9 non-tile pictures (slide regions) to paste
    train_frames = []              # (bgr, boxes) kept in memory for synthesis
    n_train = n_val = n_boxes = 0
    if args.synthetic_only:
        for old in dirs["train"][0].glob("syn_*.jpg"):
            old.unlink()
            (dirs["train"][1] / f"{old.stem}.txt").unlink(missing_ok=True)
        for img in sorted(dirs["train"][0].glob("P*.jpg")):
            bgr = cv2.imread(str(img))
            H, W = bgr.shape[:2]
            boxes = []
            for line in open(dirs["train"][1] / f"{img.stem}.txt"):
                _, cx, cy, w, h = map(float, line.split())
                boxes.append([int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)])
            train_frames.append((bgr, boxes))
            n_train += 1
            n_boxes += len(boxes)
            for x0, y0, x1, y1 in boxes:
                if len(bank) < 400 and random.random() < 0.3 and x1 - x0 >= 120:
                    bank.append(bgr[y0:y1, x0:x1].copy())
            if len(slides) < 200 and random.random() < 0.2:
                sw = int(W * 0.25)
                sh = round(sw * 9 / 16)
                for _ in range(10):
                    sx, sy = random.randint(0, W - sw), random.randint(0, H - sh)
                    if all(iou([sx, sy, sx + sw, sy + sh], b) == 0 for b in boxes):
                        region = bgr[sy:sy + sh, sx:sx + sw]
                        if region.std() > 12:
                            slides.append(region.copy())
                        break
        args.videos = []
        n_val = len(list(dirs["val"][0].glob("*.jpg")))
    for vid in args.videos:
        path = ROOT / TRUTH["files"][vid]
        if not path.exists():
            print(f"{vid}: {path} not found, skipped")
            continue
        res = regress.analysis(vid)
        labels = labels_from_timeline(res)
        truth = TRUTH["videos"][vid]
        val_t = sorted({t for role in ("child", "experimenter") for t, _ in truth[role]})
        val_boxes = {}
        for role in ("child", "experimenter"):
            for t, b in truth[role]:
                val_boxes.setdefault(t, []).append(b)
        # training moments: every `step` seconds, where the timeline has labels, away from val
        want = []
        last = -1e9
        for t in sorted(labels):
            if t - last >= args.step - 1e-6 and all(abs(t - v) > VAL_GUARD for v in val_t):
                want.append(t)
                last = t
        print(f"{vid}: {len(labels)} labelled moments, {len(want)} training frames, {len(val_t)} validation frames")
        for t, bgr in frames_at(path, want + val_t):
            H, W = bgr.shape[:2]
            if t in val_boxes:
                boxes = [[max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])] for b in val_boxes[t]]
                write_sample(*dirs["val"], f"{vid}_{t:.2f}", bgr, boxes)
                n_val += 1
            if t in labels and t in want:
                boxes = labels[t]
                write_sample(*dirs["train"], f"{vid}_{t:.2f}", bgr, boxes)
                n_train += 1
                n_boxes += len(boxes)
                train_frames.append((bgr, boxes))
                for x0, y0, x1, y1 in boxes:
                    if len(bank) < 400 and random.random() < 0.3 and x1 - x0 >= 120:
                        bank.append(bgr[y0:y1, x0:x1].copy())
                # a 16:9 region of the frame that holds no tile: slide content for the "not a tile" class
                if len(slides) < 200 and random.random() < 0.2:
                    sw = int(W * 0.25)
                    sh = round(sw * 9 / 16)
                    for _ in range(10):
                        sx, sy = random.randint(0, W - sw), random.randint(0, H - sh)
                        if all(iou([sx, sy, sx + sw, sy + sh], b) == 0 for b in boxes):
                            region = bgr[sy:sy + sh, sx:sx + sw]
                            if region.std() > 12:
                                slides.append(region.copy())
                            break
    n_syn = 0
    if bank and train_frames:
        target = int(args.synthetic * len(train_frames))
        tries = 0
        while n_syn < target and tries < 4 * target:
            tries += 1
            bgr, boxes = random.choice(train_frames)
            r = synthesize(bgr, boxes, bank, slides)
            if r is None:
                continue
            write_sample(*dirs["train"], f"syn_{n_syn:05d}", r[0], r[1])
            n_syn += 1
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: tile\n", encoding="utf-8")
    print(f"train: {n_train} real frames ({n_boxes} boxes) + {n_syn} synthetic; val: {n_val} frames; "
          f"tile bank {len(bank)}, slide bank {len(slides)}\n{out / 'data.yaml'}")


if __name__ == "__main__":
    main()
