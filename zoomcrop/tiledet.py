"""Learned tile finder: a small YOLO model that recognises camera pictures on screen.

Unlike detect.find_tiles it does not need the tile's border to be visible, so a tile
under a context menu, with the mouse over it, or with white content against a white
slide is still found. Its boxes are a few pixels loose; detect._snap pulls each side
onto the nearest real edge when there is one. The model file is produced by
tools/train.py and runs through onnxruntime (already needed for OCR); when the file is
missing, the geometric finder alone is used.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .detect import ASPECT, Tile, _snap, iou

MODEL_FILE = Path(__file__).resolve().parent / "models" / "tiles.onnx"
CONF = 0.35     # detections below this confidence are dropped
NMS_IOU = 0.5
TWIN_IOU = 0.5     # a geometric box overlapping a learned one this much is the same tile, measured exactly
PHANTOM_IOU = 0.25  # ... overlapping less than TWIN_IOU but at least this much is a phantom and is dropped


class TileDetector:
    def __init__(self, path: Path = MODEL_FILE, threads: int = 8):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        # the GPU when the installed onnxruntime offers one (onnxruntime-directml on
        # Windows: any DirectX 12 card, no CUDA needed), else the CPU
        avail = ort.get_available_providers()
        providers = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider") if p in avail]
        if "DmlExecutionProvider" in providers:
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(str(path), so, providers=providers + ["CPUExecutionProvider"])
        self.provider = self.sess.get_providers()[0]
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        self.size = int(inp.shape[2]), int(inp.shape[3])  # (h, w), fixed at export

    @staticmethod
    def available(path: Path = MODEL_FILE) -> bool:
        return path.exists()

    def detect(self, bgr: np.ndarray, gray: np.ndarray | None = None) -> list[Tile]:
        H, W = bgr.shape[:2]
        ih, iw = self.size
        r = min(ih / H, iw / W)
        nw, nh = round(W * r), round(H * r)
        canvas = np.full((ih, iw, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.name: np.ascontiguousarray(x)})[0][0]  # (4 + classes, N)
        boxes, scores = out[:4].T, out[4:].max(axis=0)
        keep = scores >= CONF
        boxes, scores = boxes[keep], scores[keep]
        if not len(boxes):
            return []
        xywh = [[float(b[0] - b[2] / 2), float(b[1] - b[3] / 2), float(b[2]), float(b[3])] for b in boxes]
        idx = cv2.dnn.NMSBoxes(xywh, scores.tolist(), CONF, NMS_IOU)
        idx = [int(i) for i in np.asarray(idx).ravel()] if len(idx) else []
        if gray is None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        tiles = []
        for i in idx:
            cx, cy, w, h = (float(v) / r for v in boxes[i])
            x0, y0, x1, y1 = int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))
            # on-screen sides are refined to the picture's real edge; a side at the frame
            # boundary is probably the tile running off-screen, and the aspect ratio then
            # says how far
            sides = "".join(s for s, on in (("t", y0 > 2), ("b", y1 < H - 2), ("l", x0 > 2), ("r", x1 < W - 2)) if on)
            x0, y0, x1, y1 = _snap(gray, (max(0, x0), max(0, y0), min(W, x1), min(H, y1)), sides, win=8)
            clip = ""
            if "t" not in sides and "b" in sides:
                y0, clip = y1 - round((x1 - x0) / ASPECT), "top"
            elif "b" not in sides and "t" in sides:
                y1, clip = y0 + round((x1 - x0) / ASPECT), "bottom"
            elif "l" not in sides and "r" in sides:
                x0, clip = x1 - round((y1 - y0) * ASPECT), "left"
            elif "r" not in sides and "l" in sides:
                x1, clip = x0 + round((y1 - y0) * ASPECT), "right"
            if clip and not ((clip == "top" and y0 < -2) or (clip == "bottom" and y1 > H + 2)
                             or (clip == "left" and x0 < -2) or (clip == "right" and x1 > W + 2)):
                clip = ""  # it just touches the frame edge
            if x1 - x0 < 24 or y1 - y0 < 14:
                continue
            tiles.append(Tile(x0, y0, x1, y1, float(scores[i]), clip, src="model"))
        return tiles


def merge(learned: list[Tile], geometric: list[Tile]) -> list[Tile]:
    """The learned boxes come first; geometric boxes that overlap none of them are kept
    (they are exact when they exist). A geometric box that agrees with a learned one
    replaces it, since its sides are measured rather than regressed. A geometric box
    that partly overlaps a learned one without being it is a phantom: a rectangle
    straddling two neighbouring tiles, or a piece of one, and is dropped."""
    out = []
    used = set()
    for tl in learned:
        # the model's box can be off by a sidebar's width (icons next to the tile get
        # included); a geometric box covering more than half of it is the same tile,
        # measured on its real edges
        twin = max((g for g in geometric if iou(g.box, tl.box) >= TWIN_IOU and g.score >= 0.6),
                   key=lambda g: iou(g.box, tl.box), default=None)
        if twin is not None:
            out.append(Tile(twin.x0, twin.y0, twin.x1, twin.y1, max(twin.score, tl.score), twin.clip or tl.clip, twin.cut))
            used.add(id(twin))
        else:
            out.append(tl)
    for g in geometric:
        if id(g) not in used and all(iou(g.box, o.box) < PHANTOM_IOU for o in out):
            out.append(g)
    return out
