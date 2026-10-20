"""Pass 1: sample the video, find tiles, follow them, and work out who is who.

Three layers, kept apart so that a mistake in one does not spread to the others:

1. Tiles and tracks. Every half second the tile finder runs on the current frame;
   a tile that keeps its place from one sample to the next (or glides, while a
   window is dragged) continues the same *track*. A track also ends when the picture
   in the window changes all at once (Zoom reshuffled who sits where). Borders that
   a single frame shows badly (a thin separator between two dark pictures) are
   recovered from an edge map averaged over the recent frames: the border stays put
   while the people move.
2. Chains and people. Tracks that cannot be the same window at once are never
   joined; tracks that show the same face, sit in the same slot, or look alike are
   chained. A chain is a person when its picture keeps changing the way live video
   does, or when a face was seen in it.
3. Names. A chain's name comes from the Zoom label when there is one, from a face
   that matches a remembered experimenter, or else "Person A/B/...". Which person
   is the child is decided afterwards (timeline.py), so changing that choice does
   not need a re-analysis.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import av
import cv2
import numpy as np

from .detect import Tile, _edges, _lines, find_tiles, iou, work_edges
from .tiledet import TileDetector, merge as merge_tiles

SAMPLE_DT = 0.5      # seconds between analysed frames
OCR_PERIOD = 5.0     # re-read an unchanged label at least this often
FULL_PERIOD = 5.0    # run the full detector at least this often, even if nothing seems to change
LEARNED_PERIOD = 0.0 # run the learned tile finder at least this often, even if the geometric one found every known
                     # window. 0 = on every full detection: skipping it when the geometric finder "agrees with
                     # itself" lets a wrong box persist (P259 6:15), and the model is what corrects it
SAME_BOX_PX = 6      # max coordinate difference (at 1080p) to treat two boxes as the same tile
THUMB_W, THUMB_H = 32, 18
SCENE_CELL = 20      # px per cell of the coarse image used to notice changes on screen
ACC_STEP = 3         # every n-th decoded frame goes into the averaged edge map
ACC_ALPHA = 0.05     # weight of a new frame in that running average (~2 s window at 10 fps)
REATTACH = 1.6       # s: a tile found again at the same place this soon continues its track
JUMP_NCC = 0.3       # thumbnail correlation below which the window shows someone else now
GLIDE_IOU = 0.6      # box overlap that still counts as the same window while it is dragged
# Recordings without name labels: a real video tile is told from slide content by its
# picture changing over time, and people are told apart by how their tile looks.
ACTIVE_LEVEL = 0.35  # mean grey-level change of a tile's thumbnail between samples that counts as
                     # "active": live video never sits perfectly still (sensor noise, small movements),
                     # slide content does, apart from the odd jump at a slide change
LIVE_MIN_ACTIVE = 6  # active pairs needed before a chain counts as live video
LIVE_RATIO = 0.5     # ... and the share of all its pairs that must be active
SAME_LOOK = 0.6      # thumbnail correlation above which two tiles show the same person
SLOT_LOOK = 0.4      # ... enough when the two tiles also sit in the same slot
TOGETHER = 0.05      # share of frames two tracks may share and still be the same person (box jitter)
FACE_SAME = 0.45     # SFace cosine similarity above which two faces are the same person
FACE_OTHER = 0.25    # ... and below which they are certainly not (SFace's own cutoff is 0.363)
FACE_MIN_VOTES = 3   # face samples that must agree before a remembered experimenter is recognised
FACE_MIN_SHARE = 0.5 # ... and the share of the person's face samples that must agree (a child's face
                     # brushes past an adult's threshold now and then; a real match does so most of the time)
UNDER_TILE = 0.15    # overlap (share of the smaller box) at which a labelled window "sits on" a box
UNDER_SHARE = 0.3    # share of a box's frames with a labelled window on it that makes it screen content
MODELS = Path(__file__).resolve().parent / "models"
OCR_PER_FRAME = 6    # at most this many label reads per analysed frame
REFINE_MAX = 6       # frames looked at between two samples while a window is being dragged
MAX_TILES = 24       # candidate boxes kept per frame (busy slides yield dozens of useless ones)
OCR_MIN_W = 0.1      # a box narrower than this share of the frame is not a video tile worth reading
MIN_SHARE = 0.03     # a person must be on screen this share of the video to be listed
MIN_SHARE_NOFACE = 0.15  # ... and more when no face was ever seen in it while others showed one
LEARNED_SHARE = 0.3  # a face-less chain counts as a person only if the learned finder saw it this often


class Cancelled(Exception):
    pass


def code_version() -> str:
    """Fingerprint of everything that shapes an analysis result: the finder code and the
    model. Analysis caches (the app's and the regression test's) are keyed by it."""
    import hashlib
    h = hashlib.sha1()
    here = Path(__file__).resolve().parent
    for f in ("analyze.py", "detect.py", "tiledet.py", "models/tiles.onnx"):
        p = here / f
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:10]


@dataclass
class Obs:
    """One tile seen in one frame."""
    box: tuple[int, int, int, int]
    score: float
    clip: str
    thumb: np.ndarray
    mask: np.ndarray
    flat: bool
    name: str = ""       # cleaned label text, "" if not read
    conf: float = 0.0
    src: str = ""        # "label": read in this frame, "carry": label unchanged since last read,
                         # "look": identified by appearance (label off-screen), "anon": by track
    hidden: bool = False  # label corner is off-screen, so the name cannot be read
    ocr_t: float = -1e9
    tb: tuple | None = None          # label text box relative to the tile's bottom-left corner
    patch: np.ndarray | None = None  # grey pixels of that text box when it was last read
    pid: int | None = None
    tid: int | None = None           # track
    t: float = 0.0
    streak: int = 0                  # consecutive samples this tile has been seen in place
    jump: bool = False               # the picture changed all at once since the previous sample
    cut: int | None = None           # bottom of the visible part when the panel edge hides the rest
    face: np.ndarray | None = None   # SFace feature of the largest face in the tile, when one was looked for
    face_q: float = 0.0              # how good that face was (detector confidence x size); -1 = none found
    learned: bool = False            # the learned tile finder saw a camera picture here (not only the geometric one)


@dataclass
class Track:
    """One window followed through consecutive samples."""
    id: int
    obs: list = field(default_factory=list)    # every observation, in time order
    keep: list = field(default_factory=list)   # a sparse sample over the track, for faces and looks
    pairs: int = 0
    active: int = 0
    ocr_fail: int = 0
    ocr_t: float = -1e9
    chain: int = -1

    @property
    def t0(self) -> float:
        return self.obs[0].t

    @property
    def t1(self) -> float:
        return self.obs[-1].t

    @property
    def live(self) -> bool:
        return self.active >= LIVE_MIN_ACTIVE and self.active >= LIVE_RATIO * self.pairs

    def names(self) -> Counter:
        return Counter(name_key(o.name) for o in self.obs if o.name)


def clean_name(text: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", text)  # "(Host)", "(Guest)" ...
    text = re.sub(r"\s+", " ", text).strip(" .,:;|·-_'\"`~()[]{}<>")
    return text


def name_key(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


def names_match(a: str, b: str, thr: float = 0.8) -> bool:
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return False
    return ka == kb or SequenceMatcher(None, ka, kb).ratio() >= thr


def _label_rect(gray, tile: Tile):
    """Pixel rectangle (y0, y1, x0, x1) of the strip where Zoom draws the name, or None."""
    H, W = gray.shape
    lh = min(int(H * 0.05), tile.h // 2)
    lw = min(tile.w, int(H * 0.35), W - tile.x0)
    if lh < 12 or lw < 24 or tile.y1 - lh < 0 or tile.y1 > H or tile.x0 < 0:
        return None
    return tile.y1 - lh, tile.y1, tile.x0, tile.x0 + lw


class FaceReader:
    """Finds the largest face in a tile and describes it, so that the same person can be
    recognised across tiles that moved, resized, or got partly covered."""

    MODEL = "sface_2021dec"  # remembered faces are only comparable within one model version

    def __init__(self):
        self.det = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (320, 180), 0.75, 0.3, 50)
        self.rec = cv2.FaceRecognizerSF.create(str(MODELS / "face_recognition_sface_2021dec.onnx"), "")

    def read(self, bgr: np.ndarray, box):
        """(feature, quality, face centre in frame coordinates) or (None, 0, None)."""
        H, W = bgr.shape[:2]
        x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
        if x1 - x0 < 60 or y1 - y0 < 30:
            return None, 0.0, None
        crop = bgr[y0:y1, x0:x1]
        scale = 480 / crop.shape[1]  # a face in a small tile is ~40 px; the detector likes it bigger
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        self.det.setInputSize((big.shape[1], big.shape[0]))
        _, faces = self.det.detect(big)
        if faces is None or len(faces) == 0:
            return None, 0.0, None
        face = max(faces, key=lambda f: f[2] * f[3])
        if face[2] / scale < 20:
            return None, 0.0, None
        feat = self.rec.feature(self.rec.alignCrop(big, face))
        centre = (x0 + (face[0] + face[2] / 2) / scale, y0 + (face[1] + face[3] / 2) / scale)
        return feat, float(face[-1]) * float(face[2] / scale), centre

    def same(self, a, b) -> float:
        return float(self.rec.match(np.asarray(a, np.float32).reshape(1, -1),
                                    np.asarray(b, np.float32).reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE))


class LabelReader:
    """Reads the Zoom name label in a tile's bottom-left corner."""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        # The detector's default upscales tiny crops to 736 px on the short side, which
        # makes a label strip take seconds; a small limit keeps it at tens of milliseconds.
        self.ocr = RapidOCR(det_limit_side_len=32, det_limit_type="min", text_score=0.3,
                            intra_op_num_threads=4, inter_op_num_threads=1)

    def read(self, gray: np.ndarray, tile: Tile):
        rect = _label_rect(gray, tile)
        if rect is None:
            return None
        H = gray.shape[0]
        y0, y1, x0, x1 = rect
        lh = y1 - y0
        crop = gray[y0:y1, x0:x1]
        if crop.std() < 6:
            return None  # blank corner (slide background): nothing to read
        scale = max(1, round(96 / lh))
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        res, _ = self.ocr(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), use_det=True, use_cls=False, use_rec=True)
        best = None
        for pts, text, conf in res or []:
            xs = [p[0] / scale for p in pts]
            ys = [p[1] / scale for p in pts]
            bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
            # Zoom draws the label at a fixed offset from the corner; text elsewhere is slide content
            if bx0 > H * 0.07 or lh - by1 > H * 0.02 or by1 - by0 < H * 0.01:
                continue
            name = clean_name(text)
            key = name_key(name)
            if len(key) < 2 and not any("一" <= ch <= "鿿" for ch in key):
                continue
            if best is None or bx0 < best[2][0]:
                tb = (int(bx0), int(by0) - lh, int(np.ceil(bx1)), int(np.ceil(by1)) - lh)
                best = (name, float(conf), tb)
        if best is None:
            return None
        name, conf, tb = best
        return name, conf, tb, _label_patch(gray, tile.box, tb)


def _label_patch(gray, box, tb, dx=0, dy=0):
    H, W = gray.shape
    x0, y0 = box[0] + tb[0] + dx, box[3] + tb[1] + dy
    x1, y1 = box[0] + tb[2] + dx, box[3] + tb[3] + dy
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 <= x0 or y1 <= y0:
        return None
    return gray[y0:y1, x0:x1].astype(np.int16)


def _label_unchanged(gray, box, prev: Obs) -> bool:
    """Is the label text still what it was when last read? Tolerates 1 px of box jitter."""
    if prev.patch is None or prev.tb is None:
        return False
    best = 255.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            p = _label_patch(gray, box, prev.tb, dx, dy)
            if p is not None and p.shape == prev.patch.shape:
                best = min(best, float(np.abs(p - prev.patch).mean()))
    return best < 12


def _thumb(gray, box):
    """Low-res copy of the tile for appearance matching, plus a mask of the part that is on screen."""
    H, W = gray.shape
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    vx0, vy0, vx1, vy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    thumb = np.zeros((THUMB_H, THUMB_W), np.float32)
    mask = np.zeros((THUMB_H, THUMB_W), bool)
    tx0, tx1 = round((vx0 - x0) / w * THUMB_W), round((vx1 - x0) / w * THUMB_W)
    ty0, ty1 = round((vy0 - y0) / h * THUMB_H), round((vy1 - y0) / h * THUMB_H)
    if tx1 > tx0 and ty1 > ty0 and vx1 > vx0 and vy1 > vy0:
        thumb[ty0:ty1, tx0:tx1] = cv2.resize(gray[vy0:vy1, vx0:vx1], (tx1 - tx0, ty1 - ty0),
                                             interpolation=cv2.INTER_AREA)
        mask[ty0:ty1, tx0:tx1] = True
    return thumb, mask


def _is_flat(gray, box) -> bool:
    """Nearly uniform picture: camera off, or a blank placeholder."""
    H, W = gray.shape
    vis = gray[max(0, box[1]):min(H, box[3]):4, max(0, box[0]):min(W, box[2]):4]
    if vis.size == 0:
        return False
    return bool((np.abs(vis.astype(np.int16) - int(np.median(vis))) <= 8).mean() > 0.9)


def _ncc(a: Obs, b: Obs) -> float:
    m = a.mask & b.mask
    if m.sum() < THUMB_W * THUMB_H * 0.15:
        return -1.0
    va, vb = a.thumb[m], b.thumb[m]
    va = va - va.mean()
    vb = vb - vb.mean()
    d = float(np.sqrt((va * va).sum() * (vb * vb).sum()))
    return float((va * vb).sum() / d) if d > 1e-6 else -1.0


def _same_box(a, b, tol) -> bool:
    return max(abs(a[i] - b[i]) for i in range(4)) <= tol


def _inside(inner, outer, tol: int = 6, ratio: float = 0.6) -> bool:
    """inner lies within outer and is clearly smaller (not just a jittered copy)."""
    if (inner[2] - inner[0]) * (inner[3] - inner[1]) >= ratio * (outer[2] - outer[0]) * (outer[3] - outer[1]):
        return False
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def _area(b) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _contains(b, pt) -> bool:
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def _takes_in(outer, inner) -> bool:
    """outer covers at least half of inner, and inner is clearly the smaller box."""
    ix = max(0, min(outer[2], inner[2]) - max(outer[0], inner[0]))
    iy = max(0, min(outer[3], inner[3]) - max(outer[1], inner[1]))
    ai = (inner[2] - inner[0]) * (inner[3] - inner[1])
    ao = (outer[2] - outer[0]) * (outer[3] - outer[1])
    return ai < 0.6 * ao and ix * iy >= 0.5 * ai


def _change(o: Obs, p: Obs, others: list[Obs]) -> float:
    """Mean change of o's thumbnail since p. Cells covered by another candidate inside
    o are left out: a slide with a video tile on it is not itself video."""
    m = o.mask & p.mask
    x0, y0, x1, y1 = o.box
    w, h = x1 - x0, y1 - y0
    for q in others:
        if q is not o and _takes_in(o.box, q.box):
            cx0 = max(0, int((q.box[0] - x0) / w * THUMB_W))
            cx1 = min(THUMB_W, int(np.ceil((q.box[2] - x0) / w * THUMB_W)))
            cy0 = max(0, int((q.box[1] - y0) / h * THUMB_H))
            cy1 = min(THUMB_H, int(np.ceil((q.box[3] - y0) / h * THUMB_H)))
            m[cy0:cy1, cx0:cx1] = False
    if m.sum() < THUMB_W * THUMB_H * 0.1:
        return 0.0
    return float(np.abs(o.thumb - p.thumb)[m].mean())


def _sides_present(gray, box, win: int = 4, need: float = 0.5) -> bool:
    """Cheap check that a known tile is still where it was: each on-screen side
    still has a straight edge along at least `need` of its length."""
    H, W = gray.shape
    x0, y0, x1, y1 = box
    m = 8
    xa, xb = max(0, x0 + m), min(W, x1 - m)
    ya, yb = max(0, y0 + m), min(H, y1 - m)
    if xb - xa < 16 or yb - ya < 16:
        return False
    for y in (y0, y1 - 1):
        if win <= y < H - win:
            if _edges(gray[y - win:y + win + 1, xa:xb])[0].mean(axis=1).max() < need:
                return False
    for x in (x0, x1 - 1):
        if win <= x < W - win:
            if _edges(gray[ya:yb, x - win:x + win + 1])[1].mean(axis=0).max() < need:
                return False
    return True


class Analyzer:
    def __init__(self, reader: LabelReader | None = None, sample_dt: float = SAMPLE_DT,
                 faces: FaceReader | None = None, known_faces: list[dict] | None = None, learned: bool = True):
        """known_faces: remembered experimenters, [{"name", "model", "vectors": [[...], ...]}]."""
        self.reader = reader or LabelReader()
        self.faces = faces or FaceReader()
        self.known = [k for k in (known_faces or []) if k.get("model") == FaceReader.MODEL and k.get("vectors")]
        # the learned tile finder, when its model has been trained (tools/train.py)
        self.tiledet = TileDetector() if learned and TileDetector.available() else None
        self.learned_t = -1e9
        self.n_learned = 0
        self.sample_dt = sample_dt
        self.best_crop: dict[str, tuple[int, bytes]] = {}
        self.tol = SAME_BOX_PX
        self.n_ocr = 0
        self.n_full = 0
        self.n_rescued = 0
        self.n_cuts = 0
        self.all_tracks: list[Track] = []  # after _identify: the tracks that were not junk
        self.junk_log: list[tuple] = []     # (loser, winner, rule) for diagnosis
        self.misses: list[tuple] = []   # (box, strip signature, t) of corners where no name was found
        self.ref_small = None           # coarse image at the last full detection
        self.ref_lines = 0              # long straight edges outside the known tiles, then
        self.ref_t = -1e9
        self.acc = None                 # running average of the edge maps (half resolution)
        self.tracks: dict[int, Track] = {}
        self.recent: dict[int, Track] = {}  # tracks seen lately (a vanished tile may come back)
        self.face_vectors: dict[int, list] = {}  # per person, for remembering experimenters (memory only)

    # -- tracks ---------------------------------------------------------------
    def _track_of(self, o: Obs) -> Track | None:
        return self.tracks.get(o.tid) if o.tid is not None else None

    def _is_live(self, o: Obs) -> bool:
        tr = self._track_of(o)
        return tr is not None and tr.live and tr.obs[-1].t - tr.obs[0].t >= 3.0

    def _relevant(self, obs: list[Obs]) -> list[Obs]:
        """Tiles that can matter for the timeline: labelled ones, ones whose label is
        off-screen, and unlabelled ones known to be live video (unless they are just a
        piece of a bigger tile)."""
        solid = [o for o in obs if o.name or self._is_live(o)]
        # unnamed tiles must have been seen in place a few times: detections that flicker
        # in and out would otherwise send the frame-by-frame search chasing them
        return [o for o in obs if o.name
                or (self._is_live(o) and o.streak >= 3
                    and not any(q is not o and _inside(o.box, q.box) for q in solid))]

    def _note(self, o: Obs, p: Obs | None, others: list[Obs]):
        """Book-keeping for a tile that has been given a track."""
        tr = self.tracks[o.tid]
        if p is not None and p.tid == o.tid and 0.2 <= o.t - p.t <= 2.0:
            tr.pairs += 1
            tr.active += _change(o, p, others) >= ACTIVE_LEVEL
        tr.obs.append(o)
        self.recent[tr.id] = tr
        if not o.clip and (not tr.keep or o.t - tr.keep[-1].t >= 2.0) and len(tr.keep) < 300:
            tr.keep.append(o)

    def _continuity(self, o: Obs, p: Obs):
        """How well o continues the window p showed a moment ago: (rank, distance, -ncc),
        smaller is better, or None. Same spot with a different picture is someone else."""
        dist = max(abs(a - b) for a, b in zip(p.box, o.box))
        if dist <= self.tol or iou(p.box, o.box) >= 0.85:
            # the same spot continues the track even when the picture jumps (a hand waved
            # in front of the camera); a jump that lasts is cut out again in _split_jumps
            return (0, dist // 3, -_ncc(o, p))
        if iou(p.box, o.box) >= GLIDE_IOU:  # being dragged
            nc = _ncc(o, p)
            return (1, 0, -nc) if nc >= 0.5 else None
        return None

    def _assign(self, out: list[Obs], prev: list[Obs], t: float):
        """Give every tile a track: the one it continues from the previous sample, else
        one that ended a moment ago at the same place, else a new one. Best matches
        first, so that a jittered duplicate box cannot steal a track from the real tile."""
        # (frames are not always analysed in time order: the frame-by-frame search
        # comes back to the span before the latest sample, hence the abs)
        self.recent = {k: tr for k, tr in self.recent.items() if abs(t - tr.obs[-1].t) <= 10}
        seen = {q.tid for q in prev}
        pool = [q for q in prev if q.tid is not None] + \
            [tr.obs[-1] for tr in self.recent.values() if tr.id not in seen and abs(t - tr.obs[-1].t) <= REATTACH]
        cands = []
        for i, o in enumerate(out):
            for j, q in enumerate(pool):
                c = self._continuity(o, q)
                if c is not None:
                    cands.append((c, i, j))
        cands.sort()
        taken_o: dict[int, Obs] = {}
        used: set[int] = set()
        for _, i, j in cands:
            q = pool[j]
            if i in taken_o or q.tid in used:
                continue
            taken_o[i] = q
            used.add(q.tid)
        for i, o in enumerate(out):
            p = taken_o.get(i)
            if p is None:
                tid = len(self.tracks)
                self.tracks[tid] = Track(tid)
            else:
                tid = p.tid
            o.tid = tid
            o.streak = p.streak + 1 if p is not None else 0
            o.jump = p is not None and _ncc(o, p) < JUMP_NCC
            self._note(o, p, out)


    # -- per frame ------------------------------------------------------------
    def _lines_outside(self, sh, sv, boxes, ds) -> int:
        """Pixels of long straight edges that do not belong to a known tile."""
        h, w = sh.shape
        min_w = max(24, int(w * 0.1))
        r = 1 if ds == 2 else 2
        top_m, bot_m, _, _ = _lines(sh, True, int(min_w * 0.8), r)
        left_m, right_m, _, _ = _lines(sv, False, int(min_w / (16 / 9) * 0.8), r)
        m = (top_m | bot_m | left_m | right_m).astype(np.uint8)
        pad = 4 * r + 4
        for x0, y0, x1, y1 in boxes:
            m[max(0, y0 // ds - pad):y1 // ds + pad, max(0, x0 // ds - pad):x1 // ds + pad] = 0
        return int(m.sum())

    def _track_only(self, gray, small, sh, sv, ds, t: float, prev: list[Obs]) -> list[Obs] | None:
        """If nothing on screen has changed since the last full detection, the
        known tiles are still valid and the detector can be skipped."""
        rel = self._relevant(prev)
        if not rel or any(o.hidden for o in rel):
            return None
        if self.ref_small is None or not (0 <= t - self.ref_t < FULL_PERIOD):
            return None
        diff = cv2.absdiff(small, self.ref_small) > 20
        for o in rel:  # the tiles themselves are live video; ignore them
            c = SCENE_CELL
            diff[max(0, o.box[1] // c - 1):o.box[3] // c + 2, max(0, o.box[0] // c - 1):o.box[2] // c + 2] = False
        if int(diff.sum()) >= 6:  # more than a mouse cursor
            return None
        # a window that appears (or comes back) brings new straight edges with it, even
        # when the picture inside holds still
        if self._lines_outside(sh, sv, [o.box for o in prev], ds) > self.ref_lines + 0.05 * sh.shape[1]:
            return None
        out = []
        for o in rel:
            if not _sides_present(gray, (o.box[0], o.box[1], o.box[2], o.cut or o.box[3])):
                return None
            thumb, mask = _thumb(gray, o.box)
            n = replace(o, thumb=thumb, mask=mask, flat=_is_flat(gray, o.box), src="carry" if o.name else "anon",
                        t=t, streak=o.streak + 1, face=None, face_q=0.0)
            # still the same person in that tile? by label if there is one, else by picture
            if o.name and not _label_unchanged(gray, o.box, o):
                return None
            if not o.name and _ncc(n, o) < 0.5:
                return None
            out.append(n)
        for n, o in zip(out, rel):
            self._note(n, o, out)
        return out

    def _strip_sig(self, gray, tile: Tile):
        rect = _label_rect(gray, tile)
        if rect is None:
            return None
        y0, y1, x0, x1 = rect
        return cv2.resize(gray[y0:y1, x0:x1], (32, 8), interpolation=cv2.INTER_AREA).astype(np.int16)

    def _detect(self, gray, edges, prev: list[Obs], frame: av.VideoFrame | None = None, t: float = 0.0) -> list[Tile]:
        """Tiles in this frame: the geometric finder's exact boxes, checked against the
        learned finder (when the model exists) whenever a known window was not found
        again, and every LEARNED_PERIOD seconds anyway, so that a window the geometric
        finder cannot see (menu over it, content touching the border) is still found
        and a phantom rectangle straddling two tiles is dropped. Borders a single frame
        shows badly are taken from the averaged edge map when the tile was there a
        moment ago."""
        tiles = find_tiles(gray, edges=edges)
        if self.tiledet is not None and frame is not None:
            known = self._relevant(prev)
            lost = any(not any(iou(p.box, tl.box) >= 0.85 for tl in tiles) for p in known)
            if lost or not known or abs(t - self.learned_t) >= LEARNED_PERIOD:
                self.learned_t = t
                self.n_learned += 1
                tiles = merge_tiles(self.tiledet.detect(frame.to_ndarray(format="bgr24"), gray), tiles)
                # A box only the model gave jitters by a few pixels from frame to frame,
                # which would break the track; when it is the known window with a small
                # shift and the same size, keep the window's box.
                for i, tl in enumerate(tiles):
                    if tl.src != "model":
                        continue
                    p = max(known, key=lambda o: iou(o.box, tl.box), default=None)
                    if p is None:
                        continue
                    pw = p.box[2] - p.box[0]
                    if iou(p.box, tl.box) >= 0.6 and abs(tl.w - pw) <= 0.1 * pw \
                            and max(abs(a - b) for a, b in zip(p.box, tl.box)) <= 0.08 * pw:
                        tiles[i] = replace(tl, x0=p.box[0], y0=p.box[1], x1=p.box[2], y1=p.box[3])
        if self.acc is not None and prev:
            lost = [p for p in self._relevant(prev) if not any(iou(p.box, tl.box) >= 0.85 for tl in tiles)]
            if lost:
                avg = (self.acc[0] >= 0.5).astype(np.uint8), (self.acc[1] >= 0.5).astype(np.uint8)
                for tl in find_tiles(gray, edges=avg):
                    if any(iou(p.box, tl.box) >= 0.85 for p in lost) and not any(iou(tl.box, k.box) >= 0.6 for k in tiles) \
                            and _sides_present(gray, (tl.x0, tl.y0, tl.x1, tl.cut or tl.y1), need=0.25):
                        tiles.append(tl)
                        self.n_rescued += 1
        return tiles

    def _frame(self, frame: av.VideoFrame, t: float, prev: list[Obs], track: bool = True) -> list[Obs]:
        gray = frame.to_ndarray(format="gray")
        H, W = gray.shape
        small = cv2.resize(gray, (W // SCENE_CELL, H // SCENE_CELL), interpolation=cv2.INTER_AREA)
        sh, sv, ds = work_edges(gray)
        if track:
            kept = self._track_only(gray, small, sh, sv, ds, t, prev)
            if kept is not None:
                self._look_for_faces(frame, kept)
                return kept
        self.n_full += 1
        out: list[Obs] = []
        tiles: list[Tile] = []
        # Two boxes with the same bottom-left corner would both read the same name label;
        # keep the one that is fully on screen, then the better supported, then the smaller.
        for tl in sorted(self._detect(gray, (sh, sv), prev, frame, t), key=lambda k: (k.clip != "", -k.score, k.w)):
            if not any(abs(tl.x0 - k.x0) <= self.tol and abs(tl.y1 - k.y1) <= self.tol for k in tiles):
                tiles.append(tl)
        # busy slides yield dozens of boxes that outscore a real tile; the ones that
        # continue a known window are kept whatever their score
        known = [o.box for o in self._relevant(prev)]
        tiles.sort(key=lambda k: (not any(iou(k.box, b) >= 0.85 for b in known), -k.score))
        tiles = tiles[:MAX_TILES]
        if track:
            self.ref_small, self.ref_t = small, t
            self.ref_lines = self._lines_outside(sh, sv, [tl.box for tl in tiles], ds)
        for tl in tiles:
            thumb, mask = _thumb(gray, tl.box)
            if tl.cut is not None:  # only the part above the cut is picture
                rows = int((tl.cut - tl.y0) / max(1, tl.h) * THUMB_H)
                mask[rows:] = False
                thumb[rows:] = 0
            o = Obs(tl.box, tl.score, tl.clip, thumb, mask, _is_flat(gray, tl.box), t=t, cut=tl.cut, learned=tl.src in ("model", "both"))
            o.hidden = tl.y1 > H or tl.x0 < 0 or tl.cut is not None
            if not o.hidden:
                p = next((q for q in prev if q.name and _same_box(q.box, tl.box, self.tol)), None)
                if p is not None and _label_unchanged(gray, tl.box, p):
                    # same label pixels as when it was last read
                    o.name, o.conf, o.tb, o.patch, o.ocr_t, o.src = p.name, p.conf, p.tb, p.patch, p.ocr_t, "carry"
                    if abs(t - p.ocr_t) >= OCR_PERIOD:
                        r = self._read(gray, tl)
                        if r:
                            o.name, o.conf, o.tb, o.patch = r
                            o.ocr_t, o.src = t, "label"
            out.append(o)
        self._assign(out, prev, t)
        # Reading labels is the slow part. A busy slide can yield a hundred box-shaped
        # scraps; only boxes big enough to be a video tile are read, best-supported first.
        budget = OCR_PER_FRAME
        for o, tl in sorted(zip(out, tiles), key=lambda ot: -ot[1].score):
            if o.name or o.hidden or tl.w < W * OCR_MIN_W or budget <= 0:
                continue
            tr = self.tracks[o.tid]
            if tr.ocr_fail >= 5 and t - tr.ocr_t < 30:
                continue  # this tile has no name label; stop asking every frame
            # Unlabelled corners are mostly slide content. Do not re-read one
            # that looks exactly as it did the last time nothing was found.
            sig = self._strip_sig(gray, tl)
            if sig is None or any(_same_box(b, tl.box, self.tol) and abs(t - mt) < 30 and np.abs(s - sig).mean() < 6
                                  for b, s, mt in self.misses):
                continue
            r = self._read(gray, tl)
            tr.ocr_t = t
            budget -= 1
            if r:
                o.name, o.conf, o.tb, o.patch = r
                o.ocr_t, o.src = t, "label"
                tr.ocr_fail = 0
                self._keep_crop(frame, tl, o.name)
            else:
                tr.ocr_fail += 1
                self.misses.append((tl.box, sig, t))
                if len(self.misses) > 400:
                    self.misses = self.misses[-200:]
        for o in out:
            if not o.name:
                o.src = "anon"
        self._look_for_faces(frame, out)
        return out

    def _look_for_faces(self, frame: av.VideoFrame, obs: list[Obs]):
        """Describe the face in every tile that just became one of its track's exemplars
        (a few times a minute per tile, so the cost stays small)."""
        bgr = None
        for o in obs:
            if o.tid is None:
                continue
            tr = self.tracks[o.tid]
            if tr.keep and tr.keep[-1] is o and o.face is None and not o.face_q:
                if bgr is None:
                    bgr = frame.to_ndarray(format="bgr24")
                o.face, o.face_q, at = self.faces.read(bgr, o.box)
                if o.face is not None and any(q is not o and _area(q.box) < _area(o.box) and _contains(q.box, at)
                                              for q in obs):
                    # the face sits in a smaller tile drawn over this box: it is that tile's
                    # face, and this box (a slide picture under the floating tiles) has none
                    o.face = None
                if o.face is None:
                    o.face_q = -1.0  # looked, found nothing

    def _read(self, gray, tl: Tile):
        self.n_ocr += 1
        return self.reader.read(gray, tl)

    def _keep_crop(self, frame: av.VideoFrame, tl: Tile, name: str):
        """Remember the largest fully visible view of each name, for the participant picker."""
        if tl.clip or tl.x0 < 0 or tl.y0 < 0 or tl.score < 0.9:
            return
        key = name_key(name)
        area = tl.w * tl.h
        if key in self.best_crop and area <= self.best_crop[key][0] * 1.2:
            return
        crop = frame.to_ndarray(format="bgr24")[tl.y0:tl.y1, tl.x0:tl.x1]
        if crop.size == 0:
            return
        if tl.w > 480:
            crop = cv2.resize(crop, (480, 270), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            self.best_crop[key] = (area, buf.tobytes())

    def _changed(self, a: list[Obs], b: list[Obs]) -> bool:
        """Did the layout change between these two frames? A window appeared or vanished,
        someone else's name is on it, or it moved by more than measurement jitter."""
        ra, rb = self._relevant(a), self._relevant(b)
        if {o.tid for o in ra} != {o.tid for o in rb}:
            return True
        at = {o.tid: o for o in rb}
        for x in ra:
            y = at[x.tid]
            if name_key(x.name) != name_key(y.name):
                return True
            slack = max(self.tol, int(0.08 * (x.box[2] - x.box[0])))
            if not _same_box(x.box, y.box, slack):
                return True
        return False

    def _refine(self, pending, a: list[Obs], b: list[Obs]):
        """The layout changed between two samples: find the exact frame.

        Binary search works for an instant switch. If a frame in between matches
        neither side (window being dragged or resized), analyse every frame.
        """
        done: dict[int, list[Obs]] = {}
        lo, hi = 0, len(pending)
        while lo < hi:
            mid = (lo + hi) // 2
            t, fr = pending[mid]
            o = done[mid] = self._frame(fr, t, a, track=False)
            if not self._changed(a, o):
                lo = mid + 1
            elif not self._changed(o, b):
                hi = mid
            else:
                # a window being dragged: follow it at a few points in between rather
                # than every frame (busy slides make each frame expensive)
                step = max(1, -(-len(pending) // REFINE_MAX))
                prev = a
                for i, (t, fr) in enumerate(pending):
                    if i % step and i not in done:
                        continue
                    if i not in done:
                        done[i] = self._frame(fr, t, prev, track=False)
                    prev = done[i]
                return [(pending[i][0], done[i]) for i in sorted(done)]
        if lo < len(pending) and lo not in done:
            done[lo] = b
        return [(pending[i][0], done[i]) for i in sorted(done)]

    def _accumulate(self, frame: av.VideoFrame, size):
        """Fold this frame's edges into the running average."""
        w, h = size
        g = frame.reformat(width=w, height=h, format="gray").to_ndarray()
        eh, ev = _edges(g)
        eh[0, :] = eh[-1, :] = 1
        ev[:, 0] = ev[:, -1] = 1
        if self.acc is None:
            self.acc = (eh.astype(np.float32), ev.astype(np.float32))
        else:
            cv2.addWeighted(self.acc[0], 1 - ACC_ALPHA, eh.astype(np.float32), ACC_ALPHA, 0, dst=self.acc[0])
            cv2.addWeighted(self.acc[1], 1 - ACC_ALPHA, ev.astype(np.float32), ACC_ALPHA, 0, dst=self.acc[1])

    # -- whole video ----------------------------------------------------------
    def run(self, path: str | Path, thumb_dir: str | Path | None = None,
            progress: Callable[[float], None] | None = None,
            cancel: Callable[[], bool] | None = None) -> dict:
        container = av.open(str(path))
        try:
            vs = container.streams.video[0]
            vs.thread_type = "AUTO"
            H, W = vs.height, vs.width
            ds = 2 if H >= 900 else 1
            acc_size = (W // ds, H // ds)
            self.tol = max(3, round(SAME_BOX_PX * H / 1080))
            fps = float(vs.average_rate or 25)
            duration = float(vs.duration * vs.time_base) if vs.duration else (container.duration or 0) / 1e6
            samples: list[tuple[float, list[Obs]]] = []
            pending: list[tuple[float, av.VideoFrame]] = []
            next_t = None
            last_t = 0.0
            n = 0
            for frame in container.decode(vs):
                t = frame.time if frame.time is not None else n / fps
                n += 1
                last_t = t
                if next_t is None:
                    next_t = t
                if n % ACC_STEP == 0:
                    self._accumulate(frame, acc_size)
                if t < next_t - 1e-6:
                    pending.append((t, frame))
                    continue
                if cancel and cancel():
                    raise Cancelled()
                prev = samples[-1][1] if samples else []
                obs = self._frame(frame, t, prev)
                if samples and pending and self._changed(prev, obs):
                    samples.extend(self._refine(pending, prev, obs))
                if prev and {o.tid for o in self._relevant(obs)} != {o.tid for o in self._relevant(prev)}:
                    self.acc = None  # new layout: start the average afresh
                samples.append((t, obs))
                pending = []
                while next_t <= t + 1e-6:
                    next_t += self.sample_dt
                if progress and duration > 0:
                    progress(min(1.0, t / duration))
            duration = max(duration, last_t + 1 / fps)
        finally:
            container.close()

        participants = self._identify(samples)
        if thumb_dir is not None:
            thumb_dir = Path(thumb_dir)
            thumb_dir.mkdir(parents=True, exist_ok=True)
        for p in participants:
            crops = [self.best_crop[k] for k in p.pop("keys") if k in self.best_crop]
            view = p.pop("view", None)
            if thumb_dir is None:
                continue
            if crops:
                (thumb_dir / f"p{p['id']}.jpg").write_bytes(max(crops, key=lambda c: c[0])[1])
                p["thumb"] = f"p{p['id']}.jpg"
            elif view is not None and _save_view(path, view.t, view.box, thumb_dir / f"p{p['id']}.jpg"):
                p["thumb"] = f"p{p['id']}.jpg"

        # Windows that show live video but could not be given to anyone go out too, marked
        # as such: the timeline uses them when the child's own window was not recognised
        # (face covered) but the other windows on screen are accounted for.
        live_tid = {tr.id: tr.live or (tr.pairs >= 2 and tr.active >= LIVE_RATIO * tr.pairs) or tr.chain >= 0
                    for tr in self.all_tracks}
        out_samples = []
        for t, obs in samples:
            known = [o for o in obs if o.pid is not None]
            extra = []
            for o in obs:
                if o.pid is not None or o.name or not live_tid.get(o.tid, False):
                    continue
                if any(_inside(o.box, q.box) or iou(o.box, q.box) >= 0.5 for q in known + extra):
                    continue  # the same window measured again, or a piece of it
                extra.append(o)
            tiles = [{"box": list(o.box), "pid": o.pid, "tid": o.tid, "src": o.src, "clip": o.clip, "cut": o.cut,
                      "flat": o.flat, "score": round(o.score, 2), "live": bool(live_tid.get(o.tid, False))}
                     for o in known + extra]
            out_samples.append({"t": round(t, 4), "tiles": tiles})
        return {
            "version": 4,
            "video": {"width": W, "height": H, "fps": fps, "duration": duration, "frames": n},
            "participants": participants,
            "samples": out_samples,
            "stats": {"ocr_calls": self.n_ocr, "full_detections": self.n_full, "learned_detections": self.n_learned,
                      "samples": len(samples), "tracks": len(self.tracks), "rescued_tiles": self.n_rescued},
        }

    # -- who is who -----------------------------------------------------------
    def _identify(self, samples) -> list[dict]:
        tracks = [tr for tr in self.tracks.values() if tr.obs]
        for tr in tracks:
            tr.obs.sort(key=lambda o: o.t)  # the frame-by-frame search adds frames out of order
            tr.keep.sort(key=lambda o: o.t)
        for tr in list(tracks):
            tracks.extend(self._split_jumps(tr))
        for tr in list(tracks):
            tracks.extend(self._split_by_face(tr))
        frame_of = {id(o): i for i, (_, obs) in enumerate(samples) for o in obs}
        frames = {tr.id: {frame_of[id(o)] for o in tr.obs if id(o) in frame_of} for tr in tracks}
        by_id = {tr.id: tr for tr in tracks}
        names = {tr.id: tr.names() for tr in tracks}

        def place_of(tr: Track):
            return tuple(int(v) for v in np.median([o.box for o in tr.obs], axis=0))

        def overlapping(a, b, thr: float = 0.4) -> bool:
            """Two boxes at the same place: similar boxes, or one mostly inside the other."""
            if iou(a, b) >= thr:
                return True
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            return ix * iy >= 0.8 * min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))

        # 1. Two tracks on screen together at largely the same place are one window
        #    measured twice. Which measurement is the window?
        #    - one box inside the other: the outer one, if it shows motion outside the
        #      inner box (a picture frame or door inside a camera view); otherwise the
        #      inner one (a box that grew into the static slide next to the tile);
        #    - boxes of about the same size: the one with the better-backed borders over
        #      its lifetime, then the longer-lived one.
        junk: set[int] = set()
        byframe: dict[int, list[Obs]] = {}
        for tr in tracks:
            for o in tr.obs:
                i = frame_of.get(id(o))
                if i is not None:
                    byframe.setdefault(i, []).append(o)
        mean_score = {tr.id: float(np.mean([o.score for o in tr.obs])) for tr in tracks}
        has_face = {tr.id: any(o.face is not None for o in tr.keep) for tr in tracks}
        solid = {tr.id for tr in tracks if tr.live or has_face[tr.id] or names[tr.id]}

        def same_size(a, b) -> bool:
            return abs((a[2] - a[0]) - (b[2] - b[0])) <= 0.06 * (b[2] - b[0]) and \
                abs((a[3] - a[1]) - (b[3] - b[1])) <= 0.06 * (b[3] - b[1])

        def peer_support(tr: Track, rival) -> float:
            """Share of tr's frames with another window of exactly the same size elsewhere
            on screen: Zoom draws the tiles of a strip or gallery all alike. Boxes inside
            the rival do not count (scraps of slide artwork come in equal sizes too)."""
            n = 0
            for o in tr.obs:
                n += any(q.tid != tr.id and q.tid in solid and q.tid not in junk and iou(o.box, q.box) < 0.1
                         and same_size(q.box, o.box) and not _takes_in(rival, q.box)
                         for q in byframe.get(frame_of.get(id(o), -1), []))
            return n / len(tr.obs)

        def weight(tr: Track) -> float:
            return len(tr.obs) * (1.5 if has_face[tr.id] else 1.0)

        # A box around a live tile that itself only "moves" where that tile is (the whole
        # shared screen, a slide region) is not a window; the face seen in it is the tile's.
        def covers(o, q) -> bool:
            return _takes_in(o.box, q.box)

        for tr in tracks:
            if tr.live:
                continue
            n = sum(any(q.tid != tr.id and q.tid in solid and covers(o, q) for q in byframe.get(frame_of.get(id(o), -1), []))
                    for o in tr.obs)
            if n >= 0.5 * len(tr.obs):
                junk.add(tr.id)

        # Zoom never draws two windows over each other. A box without a name of its own that a
        # labelled window partly covers (a picture on the slide under the floating video strip)
        # is screen content, whatever moves inside it.
        def under(o, q) -> bool:
            ix = max(0, min(o.box[2], q.box[2]) - max(o.box[0], q.box[0]))
            iy = max(0, min(o.box[3], q.box[3]) - max(o.box[1], q.box[1]))
            return ix * iy >= UNDER_TILE * min(_area(o.box), _area(q.box)) and not _takes_in(o.box, q.box)                 and not _takes_in(q.box, o.box)

        for tr in tracks:
            if tr.id in junk or names[tr.id]:
                continue
            n = sum(any(q.tid != tr.id and q.tid not in junk and names[q.tid] and under(o, q)
                        for q in byframe.get(frame_of.get(id(o), -1), []))
                    for o in tr.obs)
            if n >= UNDER_SHARE * len(tr.obs):
                junk.add(tr.id)
                self.junk_log.append((tr.id, -1, "under"))

        def outside_active(outer: Track, inner: Track) -> float:
            """Share of outer's sample pairs (with inner present in both) whose picture
            changed outside inner's box."""
            inner_at = {frame_of.get(id(o)): o for o in inner.obs}
            n = act = 0
            for p, o in zip(outer.obs, outer.obs[1:]):
                if not (0.2 <= o.t - p.t <= 2.0):
                    continue
                a, b = inner_at.get(frame_of.get(id(p))), inner_at.get(frame_of.get(id(o)))
                if a is None or b is None:
                    continue
                n += 1
                act += _change(o, p, [b]) >= ACTIVE_LEVEL
            return act / n if n else 1.0

        for tr in sorted(tracks, key=lambda tr: -len(tr.obs)):
            if tr.id in junk:
                continue
            rivals: Counter = Counter()
            for o in tr.obs:
                for q in byframe.get(frame_of.get(id(o), -1), []):
                    if q.tid != tr.id and q.tid not in junk and q.tid in by_id and overlapping(o.box, q.box):
                        rivals[q.tid] += 1
            for rid, n in rivals.most_common():
                if n < 0.6 * min(len(tr.obs), len(by_id[rid].obs)):
                    break
                other = by_id[rid]
                bo, bt = place_of(other), place_of(tr)
                st, so = peer_support(tr, bo), peer_support(other, bt)
                dur_t, dur_o = tr.t1 - tr.t0, other.t1 - other.t0
                if max(st, so) >= 0.5 and min(st, so) < 0.2:
                    loser = other if st > so else tr
                elif _inside(bt, bo, ratio=0.85):
                    # a box inside a window: content, unless the window is dead outside it
                    # and the inner box lived about as long (then the outer box is the fake)
                    loser = other if (outside_active(other, tr) < 0.3 and (tr.live or has_face[tr.id])
                                      and dur_t >= 0.5 * dur_o) else tr
                elif _inside(bo, bt, ratio=0.85):
                    loser = tr if (outside_active(tr, other) < 0.3 and (other.live or has_face[rid])
                                   and dur_o >= 0.5 * dur_t) else other
                elif abs(weight(tr) - weight(other)) > 0.3 * max(weight(tr), weight(other)):
                    loser = tr if weight(tr) < weight(other) else other
                elif abs(mean_score[tr.id] - mean_score[rid]) > 0.05:
                    loser = tr if mean_score[tr.id] < mean_score[rid] else other
                else:
                    loser = other if len(tr.obs) >= len(other.obs) else tr
                # a short overlap cannot take down a long track: the loser must spend most
                # of its life next to the winner
                if n < 0.6 * len(loser.obs):
                    continue
                junk.add(loser.id)
                self.junk_log.append((loser.id, (tr if loser is other else other).id,
                                      "size" if max(st, so) >= 0.5 and min(st, so) < 0.2 else "nested" if _inside(bt, bo, ratio=0.85) or _inside(bo, bt, ratio=0.85) else "score"))
                if loser is tr:
                    break
        tracks = [tr for tr in tracks if tr.id not in junk]
        self.all_tracks = tracks

        # 2. Chains: join tracks that are the same window seen at different times.
        #    Busy slides leave hundreds of one-off boxes; only tracks with some substance
        #    take part (the rest can still be rescued by look at the end).
        worth = [tr for tr in tracks if tr.live or names[tr.id] or len(tr.obs) >= 4
                 or any(o.face is not None for o in tr.keep)]
        ids = [tr.id for tr in sorted(worth, key=lambda tr: -len(tr.obs))[:300]]
        parent = {a: a for a in ids}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        looks = {a: (by_id[a].keep or by_id[a].obs)[::max(1, len(by_id[a].keep or by_id[a].obs) // 6)][:6] for a in ids}
        place = {a: tuple(int(v) for v in np.median([o.box for o in by_id[a].obs], axis=0)) for a in ids}
        faces = {a: sorted((o for o in by_id[a].keep if o.face is not None), key=lambda o: -o.face_q)[:8] for a in ids}

        def face_sim(a, b):
            """How alike the faces of two tracks are: each exemplar's best match on the other
            side, then the median, so one lucky pair cannot join two different people."""
            if not faces[a] or not faces[b]:
                return None
            best_a = [max(self.faces.same(p.face, q.face) for q in faces[b]) for p in faces[a]]
            best_b = [max(self.faces.same(p.face, q.face) for p in faces[a]) for q in faces[b]]
            return max(float(np.median(best_a)), float(np.median(best_b)))

        # Frames in which two tracks are on screen at different places. Two boxes on the
        # same spot in one frame are one window measured twice, not two windows.
        at = {a: {frame_of[id(o)]: o for o in by_id[a].obs if id(o) in frame_of} for a in ids}
        conflicts: dict[tuple, int] = {}

        def conflict(a, b) -> int:
            key = (a, b) if a < b else (b, a)
            if key not in conflicts:
                conflicts[key] = sum(1 for i in frames[a] & frames[b] if not overlapping(at[a][i].box, at[b][i].box, 0.1))
            return conflicts[key]

        def apart(a, b):
            """Together on screen: two windows, whatever they look like."""
            return conflict(a, b) > max(1, TOGETHER * min(len(frames[a]), len(frames[b])))

        gframes = {a: set(frames[a]) for a in ids}
        mem = {a: [a] for a in ids}

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            n = sum(conflict(x, y) for x in mem[ra] for y in mem[rb])
            if n > max(1, TOGETHER * min(len(gframes[ra]), len(gframes[rb]))):
                return
            parent[ra] = rb
            gframes[rb] |= gframes[ra]
            mem[rb] += mem[ra]

        # Strongest evidence first, so that a real person's pieces are joined before any
        # scrap of slide or picture gets attached to anything.
        proposals = []  # (rank, -strength, a, b): 0 = same name, 1 = same face, 2 = same slot, 3 = looks alike
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                a, b = ids[x], ids[y]
                if apart(a, b):
                    continue
                if names[a] and names[b]:
                    na, nb = names[a].most_common(1)[0][0], names[b].most_common(1)[0][0]
                    if names_match(na, nb):
                        proposals.append((0, 0, a, b))
                    continue  # two different labels: never the same person
                fs = face_sim(a, b)
                if fs is not None and fs >= FACE_SAME:
                    proposals.append((1, -fs, a, b))
                    continue
                if fs is not None and fs < FACE_OTHER and len(faces[a]) >= 2 and len(faces[b]) >= 2:
                    continue  # two clearly different faces: never join these, whatever the tiles look like
                gap = max(by_id[a].t0, by_id[b].t0) - min(by_id[a].t1, by_id[b].t1)
                nc = max(_ncc(p, q) for p in looks[a] for q in looks[b])
                if iou(place[a], place[b]) >= 0.7 and nc >= SLOT_LOOK and gap <= 60:
                    proposals.append((2, -iou(place[a], place[b]), a, b))
                elif nc >= SAME_LOOK and gap <= 120:
                    proposals.append((3, -nc, a, b))
        for _, _, a, b in sorted(proposals):
            union(a, b)
        groups: dict[int, list[int]] = {}
        for a in ids:
            groups.setdefault(find(a), []).append(a)

        # 3. Which chains are people. Live video keeps changing a little all the time;
        #    a face settles it too. Counts add up along the chain, so a window that moved
        #    every few seconds still qualifies.
        chains = []
        for members in groups.values():
            pairs = sum(by_id[a].pairs for a in members)
            active = sum(by_id[a].active for a in members)
            n_face = sum(len(faces[a]) for a in members)
            live = active >= LIVE_MIN_ACTIVE and active >= LIVE_RATIO * pairs
            # scraps that never held still for a few seconds are pieces of something, not a window
            settled = max(by_id[a].t1 - by_id[a].t0 for a in members) >= 3.0
            nm = Counter()
            for a in members:
                nm.update(names[a])
            views = [o for a in members for o in by_id[a].obs]
            # A chain with no face has to be a camera picture in the learned finder's eyes
            # too: slide content that moves (a reveal animation, a video on the slide) is
            # "live" to the pixels but never looks like a camera tile to the model.
            seen = self.tiledet is None or sum(1 for o in views if o.learned) >= LEARNED_SHARE * len(views)
            if (live and settled and (seen or n_face)) or n_face >= 2:
                chains.append({"members": members, "live": live, "faces": n_face, "names": nm, "views": views})
                for a in members:
                    by_id[a].chain = len(chains) - 1

        # 4. Name the chains: by label, by a remembered face, else by letter.
        dur = max(1e-6, samples[-1][0] - samples[0][0]) if samples else 1.0
        step = {i: (samples[i + 1][0] - t if i + 1 < len(samples) else self.sample_dt) for i, (t, _) in enumerate(samples)}
        for ch in chains:
            ch["share"] = sum(step[frame_of[id(o)]] for o in ch["views"] if id(o) in frame_of) / dur
            ch["vectors"] = [o.face for a in ch["members"] for o in faces[a]]
            ch["known"] = self._known_person(ch["members"], faces)
        any_face = any(ch["faces"] for ch in chains)
        chains = [ch for ch in chains if ch["share"] >= MIN_SHARE
                  and (ch["faces"] or ch["names"] or not any_face or ch["share"] >= MIN_SHARE_NOFACE)]
        # a chain whose boxes mostly sit inside another chain's window at the same time is
        # a piece of that window (camera content that formed a rectangle), not a person
        chains.sort(key=lambda c: -c["share"])
        nested = []
        for ch in chains:
            inside = 0
            for o in ch["views"]:
                fr = samples[frame_of[id(o)]][1] if id(o) in frame_of else []
                if any(q is not o and q.tid is not None and q.tid not in ch["members"] and q.tid in by_id
                       and by_id[q.tid].chain >= 0 and _inside(o.box, q.box, ratio=0.85) for q in fr):
                    inside += 1
            if inside >= 0.6 * len(ch["views"]):
                nested.append(ch)
        chains = [ch for ch in chains if ch not in nested]
        # a label seen for a sliver of the video is stray on-screen text, not a person
        for ch in chains:
            total = sum(ch["names"].values())
            if ch["names"] and total < max(3, 0.3 * len(ch["views"])):
                ch["names"] = Counter()
        # the same label in several chains (a person whose window moved beyond what the
        # chaining could follow) is one participant
        people: list[dict] = []
        for ch in sorted(chains, key=lambda c: -c["share"]):
            key = ch["names"].most_common(1)[0][0] if ch["names"] else ""
            hit = None
            if key:
                hit = next((p for p in people if p["key"] and names_match(p["key"], key)), None)
            elif ch["known"]:
                hit = next((p for p in people if p.get("known") == ch["known"]), None)
            if hit is None:
                people.append({"key": key, "known": ch["known"], "chains": [ch], "share": ch["share"]})
            else:
                hit["chains"].append(ch)
                hit["share"] += ch["share"]
        # display names: the most frequent spelling of a label
        display: dict[str, Counter] = {}
        for _, obs in samples:
            for o in obs:
                if o.name:
                    display.setdefault(name_key(o.name), Counter())[o.name] += 1
        out = []
        n_anon = 0
        for pid, p in enumerate(people):
            views = [o for ch in p["chains"] for o in ch["views"]]
            for o in views:
                o.pid = pid
                if not o.name:
                    o.src = "anon"
            keys = [k for ch in p["chains"] for k in ch["names"]]
            if p["key"]:
                nm = Counter()
                for k in keys:
                    nm.update(display.get(k, Counter()))
                name = nm.most_common(1)[0][0]
            elif p["known"]:
                name = p["known"]
            else:
                name = f"Person {chr(65 + n_anon) if n_anon < 26 else n_anon + 1}"
                n_anon += 1
            entry = {"id": pid, "name": name, "n": len(views), "share": round(min(1.0, p["share"]), 4), "keys": keys}
            if not p["key"]:
                entry["anon"] = True
                entry["view"] = self._best_view(views)
            if p["known"]:
                entry["experimenter"] = True
            vectors = [v for ch in p["chains"] for v in ch["vectors"]]
            if vectors:
                entry["face"] = True
                self.face_vectors[pid] = [np.asarray(v, np.float32).ravel().tolist() for v in vectors[:16]]
            out.append(entry)

        # 5. Remaining tiles (a window being dragged, a piece cut off by the panel edge):
        #    match by appearance against the identified views nearby in time.
        self._rescue(samples, out)
        return out

    def _known_person(self, members, faces) -> str:
        """Name of the remembered experimenter these faces belong to, or ""."""
        if not self.known:
            return ""
        votes: Counter = Counter()
        weight: Counter = Counter()
        for a in members:
            for o in faces[a]:
                best = max(((max(self.faces.same(o.face, v) for v in k["vectors"]), k["name"]) for k in self.known),
                           key=lambda x: x[0])
                if best[0] >= FACE_SAME:
                    votes[best[1]] += 1
                    weight[best[1]] += best[0] * max(0.0, o.face_q)
        if not votes:
            return ""
        name, n = votes.most_common(1)[0]
        total = sum(len(faces[a]) for a in members)
        return name if n >= FACE_MIN_VOTES and n >= FACE_MIN_SHARE * total else ""

    def _best_view(self, views):
        """The picture shown when asking who is who: a big, fully visible, well-detected
        view, preferably one with a face in it."""
        good = [o for o in views if not o.clip and o.cut is None and o.box[0] >= 0 and o.box[1] >= 0] or views
        top = sorted(good, key=lambda o: (o.score >= 0.85, (o.box[2] - o.box[0]) * (o.box[3] - o.box[1])), reverse=True)
        top = [o for o in top if (o.box[2] - o.box[0]) >= 0.8 * (top[0].box[2] - top[0].box[0])][:40]
        with_face = [o for o in top if o.face is not None]
        if with_face:
            return max(with_face, key=lambda o: o.face_q)
        return max(top, key=lambda o: float(o.thumb[o.mask].std()) if o.mask.any() else 0.0)

    def _rescue(self, samples, people):
        refs: dict[int, tuple[list[float], list[Obs]]] = {}
        for t, obs in samples:
            for o in obs:
                if o.pid is not None and not o.clip:
                    ts, os_ = refs.setdefault(o.pid, ([], []))
                    ts.append(t)
                    os_.append(o)
        anon = {p["id"] for p in people if p.get("anon")}
        for t, obs in samples:
            taken = {o.pid for o in obs if o.pid is not None}
            known = [o for o in obs if o.pid is not None]
            for o in obs:
                if o.pid is not None or o.name:
                    continue
                if any(_inside(o.box, q.box) for q in known):
                    continue  # a piece of a tile that is already identified
                scores = []
                for pid, (ts, os_) in refs.items():
                    # a named person is only matched where the label cannot be seen
                    if pid in taken or not (o.hidden or pid in anon):
                        continue
                    i = int(np.searchsorted(ts, t))
                    near = os_[max(0, i - 4):i + 4]
                    scores.append((max(_ncc(o, r) for r in near), pid))
                if not scores:
                    continue
                scores.sort(reverse=True)
                best, pid = scores[0]
                second = scores[1][0] if len(scores) > 1 else 0.0
                if best >= SAME_LOOK and best - second >= 0.15:
                    o.pid, o.src = pid, "anon" if pid in anon else "look"
                    taken.add(pid)
                    known.append(o)

    def _split_by_face(self, tr: Track) -> list[Track]:
        """One slot, several people: Zoom reuses a tile position when someone leaves or
        the active speaker changes. If the faces seen along a track fall into clearly
        different groups, cut the track where the person changed. The original keeps
        the first person's observations; the other pieces are returned as new tracks."""
        faced = [o for o in tr.keep if o.face is not None]
        if len(faced) < 4:
            return []
        groups: list[list[Obs]] = []  # exemplars with the same face
        for o in faced:
            scored = [(max(self.faces.same(o.face, q.face) for q in g), g) for g in groups]
            best = max(scored, key=lambda x: x[0]) if scored else None
            if best is not None and best[0] >= FACE_SAME:
                best[1].append(o)
            else:
                groups.append([o])
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            return []
        marks = sorted((o.t, gi) for gi, g in enumerate(groups) for o in g)
        mark_t = [m[0] for m in marks]
        views = tr.obs
        labels: list = []
        for o in views:
            k = int(np.searchsorted(mark_t, o.t))
            lo, hi = marks[max(0, k - 1)], marks[min(len(marks) - 1, k)]
            labels.append(lo[1] if lo[1] == hi[1] else None)
        i = 0
        while i < len(labels):
            if labels[i] is not None:
                i += 1
                continue
            j = i
            while j < len(labels) and labels[j] is None:
                j += 1
            left = labels[i - 1] if i > 0 else labels[j] if j < len(labels) else 0
            right = labels[j] if j < len(labels) else left
            cut, worst = i, 2.0
            for k in range(max(i, 1), min(j + 1, len(views))):
                sim = _ncc(views[k], views[k - 1])
                if sim < worst:
                    worst, cut = sim, k
            for k in range(i, j):
                labels[k] = left if k < cut else right
            i = j
        return self._cut(tr, labels)

    def _cut(self, tr: Track, labels: list) -> list[Track]:
        """Split tr into one track per label (in order of first appearance). The first
        piece keeps tr's id; the others are returned as new tracks."""
        views = tr.obs
        pieces: dict = {}
        for o, lab in zip(views, labels):
            pieces.setdefault(lab, []).append(o)
        out = []
        first = None
        for piece in pieces.values():
            if first is None:
                first = piece
                continue
            nid = 1000000 + len(self.tracks) + self.n_cuts
            self.n_cuts += 1
            nt = Track(nid)
            nt.pairs, nt.active = round(tr.pairs * len(piece) / len(views)), round(tr.active * len(piece) / len(views))
            nt.obs = piece
            nt.keep = [o for o in tr.keep if any(o is q for q in piece)] or piece[::max(1, len(piece) // 8)]
            for o in piece:
                o.tid = nid
            out.append(nt)
        tr.keep = [o for o in tr.keep if any(o is q for q in first)] or first[::max(1, len(first) // 8)]
        tr.obs = first
        tr.pairs, tr.active = round(tr.pairs * len(first) / len(views)), round(tr.active * len(first) / len(views))
        return out

    def _split_jumps(self, tr: Track) -> list[Track]:
        """Cut a track where its picture changed all at once and stayed changed: Zoom put
        someone else into that slot. A hand waved across the camera also changes
        everything for a moment, but the picture before and after still match."""
        obs = tr.obs
        labels = [0] * len(obs)
        label = 0
        for i, o in enumerate(obs):
            if o.jump and 0 < i:
                before = [q for q in obs[max(0, i - 8):i] if o.t - 2.0 <= q.t < o.t]
                after = [q for q in obs[i:i + 8] if q.t <= o.t + 2.0]
                if before and after and max(_ncc(a, b) for a in before for b in after) < JUMP_NCC:
                    label += 1
            labels[i] = label
        if label == 0:
            return []
        return self._cut(tr, labels)


def _save_view(path, t: float, box, dst: Path) -> bool:
    """Write a picture of one tile at time t (for the 'who is who' screen)."""
    try:
        with av.open(str(path)) as c:
            vs = c.streams.video[0]
            c.seek(int(max(0.0, t - 0.01) / vs.time_base), stream=vs)
            frame = next((f for f in c.decode(vs) if f.time is not None and f.time >= t - 1e-3), None)
            if frame is None:
                return False
            x0, y0, x1, y1 = box
            crop = frame.to_ndarray(format="bgr24")[max(0, y0):y1, max(0, x0):x1]
            if crop.size == 0:
                return False
            if crop.shape[1] > 480:
                crop = cv2.resize(crop, (480, round(480 * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_AREA)
            return bool(cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 85]))
    except (OSError, av.FFmpegError, StopIteration):
        return False
