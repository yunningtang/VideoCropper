"""Turn per-sample tile observations into crop segments for one participant.

A segment is a time span with one fixed crop box (or no box, when the child is
not on screen). Segments carry flags so that anything the pipeline was unsure
about can be reviewed instead of rewatching the whole video.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .analyze import names_match
from .detect import iou

GROUP_TOL = 5      # px at 1080p: box jitter below this is the same layout
FILL_SAME = 1.5    # s: bridge a detection gap this short when the box is unchanged across it
FILL_ANY = 0.6     # s: bridge a gap this short even if the box changed (avoids black flashes)
MIN_AUTO_SHARE = 0.2  # a person must be on screen this much of the video to be cropped without asking

FLAG_TEXT = {
    "missing": "Child's tile not found",
    "other": "Only other people on screen",
    "filled": "Lost briefly; previous box reused",
    "look": "Name label not visible; identified by appearance",
    "clipped": "Tile partly off-screen",
    "camera_off": "Camera may be off",
    "manual": "Set by hand",
}


def pick_child(participants: list[dict], experimenters: list[str], child_hint: str = "") -> int | None:
    """Return the participant id to crop, or None if the user has to choose."""
    if child_hint:
        hits = [p for p in participants if names_match(p["name"], child_hint)]
        if len(hits) == 1:
            return hits[0]["id"]
    # people found by look alone only matter when nobody could be read by name
    if any(not p.get("anon") for p in participants):
        participants = [p for p in participants if not p.get("anon")]
    # an experimenter is known by name (the Zoom label) or by face (remembered earlier)
    rest = [p for p in participants if not p.get("experimenter")
            and not any(names_match(p["name"], e) for e in experimenters)]
    # A "name" seen for a sliver of the video is stray on-screen text read as a label,
    # not a person. Never crop that without asking.
    if len(rest) == 1 and rest[0].get("share", 1.0) >= MIN_AUTO_SHARE:
        return rest[0]["id"]
    return None


def _fit_aspect(box, aspect=16 / 9):
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = w / aspect
    cy = (y0 + y1) / 2
    return [int(x0), int(round(cy - h / 2)), int(x1), int(round(cy - h / 2)) + int(round(h))]


def build(analysis: dict, pid: int, overrides: list[dict] | None = None) -> list[dict]:
    H = analysis["video"]["height"]
    duration = analysis["video"]["duration"]
    tol = max(3, round(GROUP_TOL * H / 1080))
    rank = {"label": 0, "carry": 0, "anon": 0, "look": 1}

    # one entry per analysed frame: the child's tile, or why there is none. When the
    # same window was measured twice in a frame, prefer the box that is the size of the
    # other people's tiles (Zoom draws them alike), then the one that continues the
    # previous choice (no twitching between two measurements), then the better-backed.
    entries = []
    prev_box = None
    for s in analysis["samples"]:
        mine = [t for t in s["tiles"] if t["pid"] == pid]
        if mine:
            peers = [t["box"] for t in s["tiles"] if t["pid"] is not None and t["pid"] != pid]

            def peer_sized(b):
                w, h = b[2] - b[0], b[3] - b[1]
                return any(abs((q[2] - q[0]) - w) <= 0.06 * w and abs((q[3] - q[1]) - h) <= 0.06 * h for q in peers)

            t = min(mine, key=lambda t: (rank.get(t["src"], 2), bool(t["clip"]), not peer_sized(t["box"]),
                                         not (prev_box is not None and iou(t["box"], prev_box) >= 0.5), -t["score"]))
            prev_box = t["box"]
            flags = set()
            if t["src"] == "look":
                flags.add("look")
            if t["clip"]:
                flags.add("clipped")
            if t["flat"]:
                flags.add("camera_off")
            if t.get("cut") is not None:
                flags.add("clipped")
            entries.append((s["t"], t["box"], flags, t.get("cut")))
        else:
            entries.append((s["t"], None, {"other"} if s["tiles"] else {"missing"}, None))

    # group consecutive entries with (nearly) the same box
    segs: list[dict] = []
    cur_boxes: list = []
    cur_cuts: list = []
    for t, box, flags, cut in entries:
        same = False
        if segs:
            last = segs[-1]
            if box is None:
                same = last["box"] is None
            elif last["box"] is not None:
                ref = np.median(np.array(cur_boxes), axis=0)
                same = bool(np.abs(ref - np.array(box)).max() <= tol)
        if same:
            segs[-1]["flags"] |= flags
            if box is not None:
                cur_boxes.append(box)
                cur_cuts.append(cut)
                segs[-1]["box"] = [int(v) for v in np.median(np.array(cur_boxes), axis=0)]
                known = [c for c in cur_cuts if c is not None]
                segs[-1]["cut"] = int(np.median(known)) if len(known) > len(cur_cuts) / 2 else None
        else:
            segs.append({"start": t, "box": box, "flags": set(flags), "cut": cut})
            cur_boxes = [box] if box is not None else []
            cur_cuts = [cut] if box is not None else []
    if segs:
        segs[0]["start"] = 0.0
    for a, b in zip(segs, segs[1:]):
        a["end"] = b["start"]
    if segs:
        segs[-1]["end"] = duration

    # bridge short gaps
    for i, s in enumerate(segs):
        if s["box"] is not None or i == 0 or segs[i - 1]["box"] is None:
            continue
        gap = s["end"] - s["start"]
        prev = segs[i - 1]
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        same_after = nxt is not None and nxt["box"] is not None and \
            max(abs(a - b) for a, b in zip(prev["box"], nxt["box"])) <= 2 * tol
        if (same_after and gap <= FILL_SAME) or (nxt is not None and nxt["box"] is not None and gap <= FILL_ANY):
            s["box"] = list(prev["box"])
            s["flags"] = {"filled"}

    segs = _merge(_settle(segs, 2 * tol))
    for s in segs:
        if s["box"] is not None:
            s["box"] = _fit_aspect(s["box"])

    for ov in overrides or []:
        segs = _apply_override(segs, ov)

    out = []
    for s in segs:
        if s["end"] - s["start"] <= 1e-6:
            continue
        flags = sorted(s["flags"])
        out.append({"start": round(s["start"], 4), "end": round(s["end"], 4), "box": s["box"], "cut": s.get("cut"),
                    "status": "ok" if s["box"] is not None else "missing", "flags": flags})
    return out


def _settle(segs, tol):
    """Neighbouring segments whose boxes differ by a few pixels are the same window
    measured slightly differently; give them the longer one's box so the output
    does not twitch."""
    # One forward pass over runs of near-identical boxes. (A loop "until nothing changes"
    # can oscillate forever when three slightly different boxes alternate in length.)
    i = 0
    while i < len(segs):
        if segs[i]["box"] is None:
            i += 1
            continue
        j = i + 1
        while j < len(segs) and segs[j]["box"] is not None and \
                max(abs(p - q) for p, q in zip(segs[j - 1]["box"], segs[j]["box"])) <= tol:
            j += 1
        run = segs[i:j]
        box = list(max(run, key=lambda s: s["end"] - s["start"])["box"])
        for s in run:
            s["box"] = box
        i = j
    return _merge(segs)


def _merge(segs):
    out: list[dict] = []
    for s in segs:
        if out and out[-1]["box"] == s["box"] and (s["box"] is None or out[-1]["flags"] == s["flags"]):
            out[-1]["end"] = s["end"]
            out[-1]["flags"] |= s["flags"]
        else:
            out.append(dict(s, flags=set(s["flags"])))
    return out


def _apply_override(segs, ov):
    a, b = float(ov["start"]), float(ov["end"])
    box = ov.get("box")
    if b <= a:
        return segs
    if ov.get("mode") == "hold":  # reuse the last box before the span
        before = [s for s in segs if s["start"] < a and s["box"] is not None]
        box = before[-1]["box"] if before else None
    elif box is not None:
        box = _fit_aspect([int(round(v)) for v in box])
    out = []
    for s in segs:
        if s["end"] <= a or s["start"] >= b:
            out.append(s)
            continue
        if s["start"] < a:
            out.append(dict(s, end=a, flags=set(s["flags"])))
        if s["end"] > b:
            out.append(dict(s, start=b, flags=set(s["flags"])))
    out.append({"start": a, "end": b, "box": box, "flags": {"manual"}})
    out.sort(key=lambda s: s["start"])
    return out


def summarize(segments: list[dict]) -> dict:
    total = sum(s["end"] - s["start"] for s in segments) or 1e-6
    visible = sum(s["end"] - s["start"] for s in segments if s["box"] is not None)
    review = [s for s in segments if s["status"] == "missing" or
              any(f in s["flags"] for f in ("filled", "look", "camera_off"))]
    boxes = [s["box"] for s in segments if s["box"] is not None]
    changes = sum(1 for a, b in zip(boxes, boxes[1:]) if a != b)
    return {
        "duration": round(total, 2),
        "visible_s": round(visible, 2),
        "visible_pct": round(100 * visible / total, 1),
        "missing_s": round(total - visible, 2),
        "review_s": round(sum(s["end"] - s["start"] for s in review), 2),
        "review_n": len(review),
        "layout_changes": changes,
    }


def write_csv(segments: list[dict], path: str | Path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["start_s", "end_s", "duration_s", "status", "x", "y", "width", "height", "flags", "notes"])
        for s in segments:
            b = s["box"]
            w.writerow([s["start"], s["end"], round(s["end"] - s["start"], 4), s["status"],
                        *(([b[0], b[1], b[2] - b[0], b[3] - b[1]]) if b else ["", "", "", ""]),
                        "|".join(s["flags"]), "; ".join(FLAG_TEXT.get(f, f) for f in s["flags"])])
