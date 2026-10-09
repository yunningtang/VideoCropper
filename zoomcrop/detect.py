"""Find Zoom video tiles (16:9 rectangles) in a screen-recording frame.

Zoom tiles are axis-aligned rectangles with a fixed 16:9 aspect ratio. We look for
long straight edges, propose rectangles from pairs of vertical sides plus one
horizontal side, and accept a proposal when enough of each side is backed by edge
pixels. Working from coverage rather than complete line segments matters because
a side is often interrupted: slide content touching the window, the mouse cursor,
or camera content that happens to match the border colour.

Three visible sides are enough when the fourth is accounted for:
* the tile is dragged partly off-screen (the box then extends beyond the frame);
* the top is under the floating window's title bar, shown while the mouse hovers.

Proposals are found on a half-resolution image (for speed) and each accepted box
is then snapped to the full-resolution edges.

None of this depends on the background colour. Slide content occasionally passes
(a box that happens to be 16:9); the name-label check in analyze.py rejects it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ASPECT = 16 / 9
EDGE_THR = 14       # grey-level step that counts as an edge (a dark border next to a black tile is ~17)
MIN_VISIBLE = 0.2   # a clipped tile must show at least this fraction of itself
COVER_4 = 0.6       # per-side edge coverage needed when all four sides are visible
COVER_3 = 0.85      # ... when one side is inferred
MAX_SEGS = 40       # longest line segments per side type used for proposals
TEXTURE_MAX = 0.3   # share of further edge structure just inside a side above which it is slide texture


@dataclass
class Tile:
    x0: int  # inner content box, x1/y1 exclusive; extends beyond the frame if clip is set
    y0: int
    x1: int
    y1: int
    score: float  # edge coverage of the weakest verified side, 0..1
    clip: str = ""  # "", "bottom", "top", "left", "right"
    cut: int | None = None  # the floating panel ends here, hiding the rest of the tile (a y value)
    src: str = "geo"  # "geo": measured from edges; "model": the learned finder's box, a few px loose

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def _inside(inner, outer, tol: int = 2) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def _rect(w, h):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))


def _edges(gray: np.ndarray):
    """(horizontal-edge mask, vertical-edge mask). 2-pixel differences, so that
    blurred edges still clear the threshold; an edge between rows k and k+1 marks both."""
    H, W = gray.shape
    eh = np.zeros((H, W), np.uint8)
    ev = np.zeros((H, W), np.uint8)
    if H > 2 and W > 2:
        cv2.threshold(cv2.absdiff(gray[2:, :], gray[:-2, :]), EDGE_THR, 1, cv2.THRESH_BINARY, dst=eh[1:-1, :])
        cv2.threshold(cv2.absdiff(gray[:, 2:], gray[:, :-2]), EDGE_THR, 1, cv2.THRESH_BINARY, dst=ev[:, 1:-1])
    return eh, ev


def _frame_edges(eh, ev):
    """The picture's own border counts as an edge, so a tile pushed flush against the
    screen edge still has four sides."""
    eh[0, :] = eh[-1, :] = 1
    ev[:, 0] = ev[:, -1] = 1


def _lines(edges: np.ndarray, horizontal: bool, min_len: int, r: int):
    """Long straight edge structures and their two boundaries.

    Returns (far_mask, near_mask, far_segs, near_segs). For horizontal lines
    "far" is the lower boundary of each structure (a tile's top side: content
    starts below it) and "near" the upper boundary (a tile's bottom side). For
    vertical lines far is the right boundary (a tile's left side) and near the
    left one. Segments are (pos, a, b): boundary row/column and extent, longest
    first. Masks mark all accepted segments, widened across the line so coverage
    can be read off a single row/column with a little tolerance.
    """
    def k(across, along):
        return _rect(along, across) if horizontal else _rect(across, along)

    m = cv2.dilate(edges, k(2 * r + 1, 1))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k(1, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k(1, min_len))
    far = np.zeros_like(m)
    near = np.zeros_like(m)
    if horizontal:
        np.greater(m[:-1], m[1:], out=far[:-1].view(bool))
        np.greater(m[1:], m[:-1], out=near[1:].view(bool))
    else:
        np.greater(m[:, :-1], m[:, 1:], out=far[:, :-1].view(bool))
        np.greater(m[:, 1:], m[:, :-1], out=near[:, 1:].view(bool))
    # Thin structures also count as side evidence where they stand: camera content
    # with an edge right next to the border (a curtain fold, a door frame) shifts the
    # structure's boundary off the border, but the structure still covers the border.
    # Thick structures are texture and count only through their boundary.
    thin = m & ~cv2.morphologyEx(m, cv2.MORPH_OPEN, k(4 * r + 9, 1))
    out = []
    for b, sign in ((far, 1), (near, -1)):
        s = cv2.dilate(b, k(3, 1))
        s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, k(1, 3))
        s = cv2.morphologyEx(s, cv2.MORPH_OPEN, k(1, min_len))  # straight runs only
        n, _, stats, _ = cv2.connectedComponentsWithStats(s, connectivity=8)
        length = stats[:, cv2.CC_STAT_WIDTH if horizontal else cv2.CC_STAT_HEIGHT]
        order = np.argsort(-length[1:]) + 1
        segs = []
        mask = np.zeros_like(m)
        for i in order:
            x, y, w, h = (int(v) for v in stats[i, :4])
            pos = y + h // 2 if horizontal else x + w // 2
            # A tile's side has plain camera picture just inside it. Inside busy
            # slide artwork the next edge structure follows within a few pixels;
            # dropping those keeps textured slides from flooding the proposals.
            lo, hi = sorted((pos + sign * 2, pos + sign * (2 * r + 4)))
            strip = m[max(0, lo):hi, x:x + w] if horizontal else m[y:y + h, max(0, lo):hi]
            if strip.size and strip.mean() > TEXTURE_MAX:
                continue
            if horizontal:
                mask[max(0, y - 1):y + h + 1, x:x + w] = 1
            else:
                mask[y:y + h, max(0, x - 1):x + w + 1] = 1
            if len(segs) < MAX_SEGS:
                segs.append((pos, x, x + w) if horizontal else (pos, y, y + h))
        # line the thin structures up with where a boundary at the same border would be
        d = r + 1
        if horizontal:
            if sign > 0:
                mask[d:] |= thin[:-d]
            else:
                mask[:-d] |= thin[d:]
        elif sign > 0:
            mask[:, d:] |= thin[:, :-d]
        else:
            mask[:, :-d] |= thin[:, d:]
        out.append((mask, segs))
    return out[0][0], out[1][0], out[0][1], out[1][1]


def _dark_strip(gray: np.ndarray, x0: int, x1: int, ya: int, yb: int) -> bool:
    """Are rows ya..yb a uniform dark strip (the floating window's title bar)?"""
    H, W = gray.shape
    x0, x1 = max(0, x0), min(W, x1)
    ya, yb = max(0, ya), min(H, yb)
    if yb - ya < 3 or x1 - x0 < 16:
        return False
    px = gray[ya:yb:2, x0:x1:4]
    if px.mean() > 80:
        return False
    q25, q50, q75 = np.percentile(px, (25, 50, 75))
    return bool(q50 <= 70 and q75 - q25 <= 8)


def _title_bar_end(gray: np.ndarray, x0: int, x1: int, y0: int, max_h: int) -> int:
    """If rows starting at y0 are a title bar, return the first row below it; otherwise y0."""
    H = gray.shape[0]
    y = y0
    while y < min(H, y0 + max_h) and _dark_strip(gray, x0, x1, y, y + 4):
        y += 2
    y += 3 if y > y0 else 0
    return y if 8 <= y - y0 < max_h else y0


def _snap(gray, box, sides: str, win: int = 5):
    """Move the named sides ("t", "b", "l", "r") of a coarse box onto the innermost
    full-resolution edge within win pixels."""
    H, W = gray.shape
    x0, y0, x1, y1 = box
    xa, xb = max(0, x0 + 2 * win), min(W, x1 - 2 * win)
    ya, yb = max(0, y0 + 2 * win), min(H, y1 - 2 * win)

    def pick(cov, innermost_last):
        # only move onto a clear edge; inside the window, prefer the innermost strong one
        if cov.size == 0 or cov.max() < 0.5:
            return None
        idx = np.nonzero(cov >= 0.6 * cov.max())[0]
        return int(idx[-1] if innermost_last else idx[0])

    def rows(center):
        lo, hi = max(0, center - win), min(H, center + win + 1)
        return lo, _edges(gray[lo:hi, xa:xb])[0].mean(axis=1) if hi - lo > 2 else np.zeros(0)

    def cols(center):
        lo, hi = max(0, center - win), min(W, center + win + 1)
        return lo, _edges(gray[ya:yb, lo:hi])[1].mean(axis=0) if hi - lo > 2 else np.zeros(0)

    if xb - xa > 8:
        if "t" in sides:
            lo, cov = rows(y0)
            i = pick(cov, True)
            if i is not None:
                y0 = lo + i
        if "b" in sides:
            lo, cov = rows(y1 - 1)
            i = pick(cov, False)
            if i is not None:
                y1 = lo + i + 1
    if yb - ya > 8:
        if "l" in sides:
            lo, cov = cols(x0)
            i = pick(cov, True)
            if i is not None:
                x0 = lo + i
        if "r" in sides:
            lo, cov = cols(x1 - 1)
            i = pick(cov, False)
            if i is not None:
                x1 = lo + i + 1
    return x0, y0, x1, y1


def work_scale(H: int) -> int:
    """The finder works on a half-size image for HD frames."""
    return 2 if H >= 900 else 1


def work_edges(gray: np.ndarray):
    """(horizontal edges, vertical edges, scale) at the finder's working size, with the
    frame border counted as an edge. Callers that average edge maps over time pass
    the result back through find_tiles(edges=...)."""
    H, W = gray.shape
    ds = work_scale(H)
    small = cv2.resize(gray, (W // 2, H // 2), interpolation=cv2.INTER_AREA) if ds == 2 else gray
    sh, sv = _edges(small)
    _frame_edges(sh, sv)
    return sh, sv, ds


def find_tiles(gray: np.ndarray, min_w_frac: float = 0.1, edges=None) -> list[Tile]:
    H, W = gray.shape
    if edges is None:
        sh, sv, ds = work_edges(gray)
    else:
        sh, sv = edges
        ds = work_scale(H)
    h, w = sh.shape
    r = 1 if ds == 2 else 2   # edges this close count as one line (a border is two edges)

    min_w = max(24, int(w * min_w_frac))
    min_h = int(min_w / ASPECT)
    tol = max(2, int(w * 0.004))
    bar_h = max(16, int(H * 0.06))
    top_m, bot_m, top_s, bot_s = _lines(sh, True, int(min_w * 0.8), r)
    left_m, right_m, left_s, right_s = _lines(sv, False, int(min_h * 0.8), r)

    # All geometry below is in the small image's coordinates until the snap step.
    # A structure's boundary sits r beyond the edge pixels, hence the offsets.
    # A side is backed by its inner boundary, or by either boundary up to a border's
    # width further out: when camera content touches the border from inside, the inner
    # edge gets swallowed, but the outer edge (slide -> border) is still clean.
    any_h = top_m | bot_m
    any_v = left_m | right_m
    bw_px = 2 * r + 3  # border width plus the shift content merging into it can cause, in small pixels

    # Coverage of a side = share of its length backed by edge pixels, read off running
    # sums so that the thousands of proposals below cost a subtraction each. The "any"
    # masks are first widened towards the outside of the tile (max over bw_px rows or
    # columns), which is where the outer edge of a border sits.
    def run_x(m):  # sums along rows: sum(m[y, a:b]) = cs[y, b] - cs[y, a]
        cs = np.zeros((h, w + 1), np.int32)
        np.cumsum(m, axis=1, out=cs[:, 1:])
        return cs

    def run_y(m):  # sums along columns
        cs = np.zeros((h + 1, w), np.int32)
        np.cumsum(m, axis=0, out=cs[1:, :])
        return cs

    k_up = cv2.dilate(any_h, _rect(1, bw_px + 1), anchor=(0, bw_px))      # max over rows p-bw_px..p
    k_down = cv2.dilate(any_h, _rect(1, bw_px + 1), anchor=(0, 0))        # rows p..p+bw_px
    k_left = cv2.dilate(any_v, _rect(bw_px + 1, 1), anchor=(bw_px, 0))    # columns p-bw_px..p
    k_right = cv2.dilate(any_v, _rect(bw_px + 1, 1), anchor=(0, 0))       # columns p..p+bw_px
    cs_top, cs_up, cs_bot, cs_down = run_x(top_m), run_x(k_up), run_x(bot_m), run_x(k_down)
    cs_left, cs_kl, cs_right, cs_kr = run_y(left_m), run_y(k_left), run_y(right_m), run_y(k_right)
    cs_any_v = run_y(any_v)

    def cov_top(y0, xa, xb):
        p, xa, xb = y0 + r, max(0, xa), min(w, xb)
        if not (0 <= p < h and xb > xa):
            return 0.0
        return max(int(cs_top[p, xb]) - int(cs_top[p, xa]), int(cs_up[p, xb]) - int(cs_up[p, xa])) / (xb - xa)

    def cov_bot(y1, xa, xb):
        p, xa, xb = y1 - r - 1, max(0, xa), min(w, xb)
        if not (0 <= p < h and xb > xa):
            return 0.0
        return max(int(cs_bot[p, xb]) - int(cs_bot[p, xa]), int(cs_down[p, xb]) - int(cs_down[p, xa])) / (xb - xa)

    def cov_left(x0, ya, yb):
        p, ya, yb = x0 + r, max(0, ya), min(h, yb)
        if not (0 <= p < w and yb > ya):
            return 0.0
        return max(int(cs_left[yb, p]) - int(cs_left[ya, p]), int(cs_kl[yb, p]) - int(cs_kl[ya, p])) / (yb - ya)

    def cov_right(x1, ya, yb):
        p, ya, yb = x1 - r - 1, max(0, ya), min(h, yb)
        if not (0 <= p < w and yb > ya):
            return 0.0
        return max(int(cs_right[yb, p]) - int(cs_right[ya, p]), int(cs_kr[yb, p]) - int(cs_kr[ya, p])) / (yb - ya)

    tops = [(p - r, a, b) for p, a, b in top_s]          # (y0, xa, xb)
    bots = [(p + r + 1, a, b) for p, a, b in bot_s]      # (y1, xa, xb)
    lefts = [(p - r, a, b) for p, a, b in left_s]        # (x0, ya, yb)
    rights = [(p + r + 1, a, b) for p, a, b in right_s]  # (x1, ya, yb)
    edge = r + 2  # a side this close to the frame boundary cannot be verified

    found: list[tuple] = []  # (score, x0, y0, x1, y1, sides to snap)

    def try_box(x0, x1, y0, y1, anchor):
        """Verify a proposal whose vertical sides are both on screen."""
        side = min(cov_left(x0, y0, y1), cov_right(x1, y0, y1))
        if side < COVER_4:
            return
        top = cov_top(y0, x0, x1)
        bot = cov_bot(y1, x0, x1)
        if min(top, bot) >= COVER_4:
            found.append((min(side, top, bot), x0, y0, x1, y1, "tblr"))
        elif side >= COVER_3:
            if anchor == "top" and top >= COVER_3:
                # bottom off-screen, or simply invisible (a white object held up in front
                # of the camera against a white slide): the sides run the full height
                found.append((min(side, top) * (0.9 if y1 >= h - edge else 0.85), x0, y0, x1, y1, "tlr"))
            elif anchor == "bottom" and bot >= COVER_3 and (
                    y0 <= edge or _dark_strip(gray, x0 * ds, x1 * ds, y0 * ds - 12, y0 * ds - 4)):
                found.append((min(side, bot) * 0.95, x0, y0, x1, y1, "blr"))

    # both vertical sides visible
    for (x0, la, lb) in lefts:
        for (x1, ra, rb) in rights:
            bw = x1 - x0
            if bw < min_w:
                continue
            bh = round(bw / ASPECT)
            ya, yb = max(la, ra), min(lb, rb)  # rows where both sides exist
            if yb - ya < 0.4 * min(bh, h):
                continue
            for (y0, a, b) in tops:
                if min(b, x1) - max(a, x0) < 0.3 * bw or not (ya - 0.6 * bh <= y0 <= yb - 0.3 * bh):
                    continue
                yt = y0
                if (y0 + bh) * ds + bar_h > H:  # bottom may be off-screen with the title bar showing
                    yt = _title_bar_end(gray, x0 * ds, x1 * ds, y0 * ds, bar_h) // ds
                if yt == y0:
                    try_box(x0, x1, y0, y0 + bh, "top")
                elif yt + bh >= h - edge and min(cov_left(x0, yt, h), cov_right(x1, yt, h)) >= COVER_3:
                    found.append((0.85, x0, yt, x1, yt + bh, "lr"))
            for (y1, a, b) in bots:
                if min(b, x1) - max(a, x0) < 0.3 * bw or not (ya + 0.3 * bh <= y1 <= yb + 0.6 * bh):
                    continue
                try_box(x0, x1, y1 - bh, y1, "bottom")

    # Second route: a top/bottom pair, with the vertical sides read straight off the
    # edge masks between them. This does not depend on the side segments, which get
    # dropped when camera content (a door frame, a bookshelf) sits right behind the
    # border. A side may also be missing altogether: off-screen, or weakly marked; the
    # aspect ratio then says where it is.
    def split_between(y0, y1, xa, xb):
        """Is there a horizontal line across this span? Then it is two stacked tiles, not one."""
        return any(y0 + min_h // 2 < y < y1 - min_h // 2 and min(b, xb) - max(a, xa) >= 0.5 * (xb - xa)
                   for y, a, b in tops + bots)

    def side_peaks(y0, y1, xa, xb):
        """Columns between xa and xb where a vertical edge runs along most of y0..y1."""
        xa, xb = max(0, xa), min(w, xb)
        if xb - xa < 4 or y1 - y0 < 4:
            return []
        ya, yb = max(0, y0), min(h, y1)
        prof = (cs_any_v[yb, xa:xb] - cs_any_v[ya, xa:xb]) / (yb - ya)
        on = np.nonzero(prof >= COVER_4)[0]
        peaks, start = [], None
        for i, x in enumerate(on):
            if start is None:
                start = x
            if i + 1 == len(on) or on[i + 1] != x + 1:
                # a border next to an edge inside the picture forms one wide run: offer its
                # strongest column and both ends, the checks below sort out which is real
                seg = prof[start:x + 1]
                for c in {int(start), int(x), int(start + int(np.argmax(seg)))}:
                    peaks.append((c + xa, float(prof[c])))
                start = None
        return peaks

    seen_pairs = set()
    for (y0, ta, tb) in tops:
        for (y1, ba, bb) in bots:
            xa, xb = max(ta, ba), min(tb, bb)
            bh = y1 - y0
            if bh < min_h or xb - xa < min_w or split_between(y0, y1, xa, xb):
                continue
            yt = _title_bar_end(gray, max(0, xa * ds), min(W, xb * ds), y0 * ds, bar_h) // ds
            bw = round((y1 - yt) * ASPECT)
            peaks = side_peaks(yt, y1, xa - 3 * tol, xb + 3 * tol)
            for (px, _) in peaks:
                for x0, x1 in ((px - r, px - r + bw), (px + r + 1 - bw, px + r + 1)):
                    key = (x0, yt, x1, y1)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    c = min(cov_bot(y1, x0, x1), cov_top(y0, x0, x1))
                    if c < COVER_4:
                        continue
                    left, right = cov_left(x0, yt, y1), cov_right(x1, yt, y1)
                    sides = "tblr" if yt == y0 else "blr"
                    if min(left, right) >= COVER_4:
                        found.append((min(c, left, right) * (1 if yt == y0 else 0.95), x0, yt, x1, y1, sides))
                    elif c >= COVER_3 and max(left, right) >= COVER_3 and (
                            (left < right and (x0 <= edge or left >= 0.2)) or
                            (right <= left and (x1 >= w - edge or right >= 0.2))):
                        strong = "r" if left < right else "l"
                        found.append((min(c, max(left, right)) * 0.9, x0, yt, x1, y1, sides.replace("lr", strong)))
            # the same with the side segments, which are placed more precisely when they exist
            for (x0, la, lb) in lefts:
                if abs(x0 - xa) > 3 * tol:
                    continue
                x1 = x0 + bw
                c = min(cov_left(x0, yt, y1), cov_bot(y1, x0, x1), cov_top(y0, x0, x1))
                if c >= COVER_3 and (x1 >= w - edge or cov_right(x1, yt, y1) >= 0.2):
                    found.append((c * 0.9, x0, yt, x1, y1, "bl" if yt != y0 else "tbl"))
            for (x1, ra, rb) in rights:
                if abs(x1 - xb) > 3 * tol:
                    continue
                x0 = x1 - bw
                c = min(cov_right(x1, yt, y1), cov_bot(y1, x0, x1), cov_top(y0, x0, x1))
                if c >= COVER_3 and (x0 <= edge or cov_left(x0, yt, y1) >= 0.2):
                    found.append((c * 0.9, x0, yt, x1, y1, "br" if yt != y0 else "tbr"))

    # best first, drop overlaps, then snap survivors to the full-resolution edges
    found.sort(key=lambda f: -f[0])
    kept: list[tuple] = []
    for f in found:
        clash = next((k for k in kept if iou(f[1:5], k[1:5]) >= 0.6), None)
        if clash is None:
            kept.append(f)
        elif f[0] >= 0.8 and _inside(clash[1:5], f[1:5]) and                 sum(abs(a - b) <= 2 for a, b in zip(clash[1:5], f[1:5])) >= 2:
            # an edge inside the picture cuts a tile short on one side; the box that
            # shares the other sides and reaches further is the real border
            kept[kept.index(clash)] = f
        elif f[5] == "tblr" and clash[5] != "tblr" and f[0] >= COVER_4:
            # four verified sides beat a box with a guessed side, whatever the scores say
            kept[kept.index(clash)] = f

    # Tiles come in strips. Next to an accepted tile there may be another whose shared
    # side is weak (two dark pictures meeting at a thin separator) and whose far side is
    # off-screen or hidden; its two other sides still have to run the full length.
    def neighbour(k):
        _, x0, y0, x1, y1, _ = k[:6]
        bw, bh = x1 - x0, y1 - y0
        out = []
        ext = max(2, bh // 5)  # how far past the far side the long sides are checked to stop
        def side_l(xx, ya, yb):
            return 1.0 if xx <= 1 else cov_left(xx, ya, yb)

        def side_r(xx, ya, yb):
            return 1.0 if xx >= w - 1 else cov_right(xx, ya, yb)

        for (nx0, ny0, nx1, ny1, strong, shared, far, beyond) in (
                (x0, y1, x1, y1 + bh, lambda: min(side_l(x0, y1, y1 + bh), side_r(x1, y1, y1 + bh)),
                 lambda: cov_top(y1, x0, x1), lambda: cov_bot(y1 + bh, x0, x1) if y1 + bh < h - edge else 1.0,
                 lambda: max(cov_left(x0, y1 + bh, y1 + bh + ext) if x0 > edge else 0.0,
                             cov_right(x1, y1 + bh, y1 + bh + ext) if x1 < w - edge else 0.0)),
                (x0, y0 - bh, x1, y0, lambda: min(side_l(x0, y0 - bh, y0), side_r(x1, y0 - bh, y0)),
                 lambda: cov_bot(y0, x0, x1), lambda: cov_top(y0 - bh, x0, x1) if y0 - bh > edge else 1.0,
                 lambda: max(cov_left(x0, y0 - bh - ext, y0 - bh) if x0 > edge else 0.0,
                             cov_right(x1, y0 - bh - ext, y0 - bh) if x1 < w - edge else 0.0)),
                (x1, y0, x1 + bw, y1, lambda: min(cov_top(y0, x1, x1 + bw), cov_bot(y1, x1, x1 + bw)),
                 lambda: cov_left(x1, y0, y1), lambda: cov_right(x1 + bw, y0, y1) if x1 + bw < w - edge else 1.0,
                 lambda: max(cov_top(y0, x1 + bw, x1 + bw + ext) if y0 > edge else 0.0,
                             cov_bot(y1, x1 + bw, x1 + bw + ext) if y1 < h - edge else 0.0)),
                (x0 - bw, y0, x0, y1, lambda: min(cov_top(y0, x0 - bw, x0), cov_bot(y1, x0 - bw, x0)),
                 lambda: cov_right(x0, y0, y1), lambda: cov_left(x0 - bw, y0, y1) if x0 - bw > edge else 1.0,
                 lambda: max(cov_top(y0, x0 - bw - ext, x0 - bw) if y0 > edge else 0.0,
                             cov_bot(y1, x0 - bw - ext, x0 - bw) if y1 < h - edge else 0.0))):
            vis_w, vis_h = min(w, nx1) - max(0, nx0), min(h, ny1) - max(0, ny0)
            if vis_w < 0.5 * bw or vis_h < 0.5 * bh:
                continue
            st = strong()
            sh = shared()
            if st >= 0.65 and sh >= 0.4 and far() >= 0.6:
                out.append((0.6, nx0, ny0, nx1, ny1, "tblr"))
                continue
            if st < COVER_3 or sh < 0.2:
                continue
            if far() >= 0.2:
                out.append((0.8, nx0, ny0, nx1, ny1, "tblr"))
            elif st >= 0.95 and beyond() < 0.5:
                # the far side is invisible (something white held up against a white slide),
                # but both long sides run exactly one tile and then stop
                out.append((0.75, nx0, ny0, nx1, ny1, "tblr"))
        return out

    for k in list(kept):
        for f in neighbour(k):
            if all(iou(f[1:5], q[1:5]) < 0.6 for q in kept):
                kept.append(f)

    # A box whose left or right side was guessed must not start inside a verified tile on
    # the same rows: that is an edge inside the picture posing as the border of a
    # phantom neighbour running off the screen.
    solid = [k for k in kept if k[5] == "tblr"]
    kept = [f for f in kept if ("l" in f[5] and "r" in f[5]) or not any(
        k is not f and abs(k[2] - f[2]) <= tol and abs(k[4] - f[4]) <= tol
        and (k[1] + tol < f[1] < k[3] - tol or k[1] + tol < f[3] < k[3] - tol) for k in solid)]
    tol_f = tol * ds
    tiles: list[Tile] = []
    for score, x0, y0, x1, y1, sides, *rest in kept:
        cut = rest[0] * ds if rest else None
        x0, y0, x1, y1 = (v * ds for v in (x0, y0, x1, y1))
        if ds > 1:
            x0, y0, x1, y1 = _snap(gray, (x0, y0, x1, y1), sides)
        # the side that was not seen follows from the aspect ratio
        if "t" not in sides or "b" not in sides:
            bh = round((x1 - x0) / ASPECT)
            if "b" in sides:
                y0 = y1 - bh
            else:
                y1 = y0 + bh
        elif "l" not in sides or "r" not in sides:
            bw = round((y1 - y0) * ASPECT)
            if "r" in sides:
                x0 = x1 - bw
            else:
                x1 = x0 + bw
        bw, bh = x1 - x0, y1 - y0
        if bw < min_w * ds or bh <= 0:
            continue
        vis = max(0, min(W, x1) - max(0, x0)) * max(0, min(H, y1) - max(0, y0)) / (bw * bh)
        if vis < MIN_VISIBLE or bw * bh >= 0.9 * W * H:
            continue
        clip = "top" if y0 < -tol_f else "bottom" if y1 > H + tol_f else \
            "left" if x0 < -tol_f else "right" if x1 > W + tol_f else ""
        tiles.append(Tile(int(x0), int(y0), int(x1), int(y1), float(score), clip, int(cut) if cut else None))
    return tiles
