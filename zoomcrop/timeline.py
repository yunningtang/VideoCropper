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
SAME_IOU = 0.85    # ... or boxes overlapping this much: two measurements of one window (loose model box, snapped edges)
FILL_SAME = 1.5    # s: bridge a detection gap this short when the box is unchanged across it
FILL_ANY = 0.6     # s: bridge a gap this short even if the box changed (avoids black flashes)
MIN_AUTO_SHARE = 0.2  # a person must be on screen this much of the video to be cropped without asking
CONTINUE_GAP = 3.0    # s: an unrecognised window still counts as continuing the child's last one this long after it

FLAG_TEXT = {
    "missing": "Child's tile not found",
    "other": "Other windows on screen, none of them the child's",
    "filled": "Lost briefly; previous box reused",
    "bridged": "Lost for a moment; same box before and after",
    "inferred": "Child not recognised in the tile; it is the window left after the experimenter's",
    "held": "Window at the child's previous place; nobody on screen was recognised",
    "unsure": "A video window is on screen but could not be told apart from the experimenter's",
    "look": "Name label not visible; identified by appearance",
    "clipped": "Tile partly off-screen",
    "camera_off": "Camera may be off",
    "manual": "Set by hand",
    "checked": "Looked at and accepted",
}
REVIEW_FLAGS = ("held", "unsure", "look", "camera_off")  # "filled" (a sub-second gap bridged with the box before it) is noted, not reviewed
MIN_REVIEW = 0.5  # s: doubtful spans shorter than this, in a row, are not worth a look
SPLIT_FLAGS = {"held", "unsure", "inferred", "look", "camera_off"}  # a segment holds one of these states, not a mix

NOTICE_TEXT = {
    "no_experimenter": "Nobody in this video was recognised as an experimenter, so the only person found "
                       "was taken to be the child. Check that the cropped window is the child's.",
    "extra_people": "Other short-lived people were found and ignored. Check the spans where the child "
                    "is marked as not on screen.",
}


def is_experimenter(p: dict, experimenters: list[str]) -> bool:
    """Known by face (remembered earlier) or by name (the Zoom label)."""
    return bool(p.get("experimenter")) or any(names_match(p["name"], e) for e in experimenters)


def decide(participants: list[dict], experimenters: list[str], child_hint: str = "") -> dict:
    """Who to crop: {"pid": id or None, "notice": why the choice should be checked}.

    Rule: take the experimenters out; whoever is left is the child. The user is only
    asked when more than one substantial person remains (a third window), or when
    nobody substantial does. One person left with no experimenter recognised at all
    is cropped but flagged, since nothing says the single window is not the
    experimenter's."""
    if child_hint:
        hits = [p for p in participants if names_match(p["name"], child_hint)]
        if len(hits) == 1:
            return {"pid": hits[0]["id"], "notice": ""}
    exps = [p for p in participants if is_experimenter(p, experimenters)]
    rest = [p for p in participants if not is_experimenter(p, experimenters)]
    # A "person" seen for a sliver of the video with no face is stray slide content or
    # a scrap of someone's window; a face makes even a brief appearance a real person.
    # And once faces were seen at all, a face-less "person" of any length is slide
    # content that moves (animations, a video on the slide): it is still offered on the
    # picker, but it does not count as a third person that forces the question.
    any_face = any(p.get("face") for p in participants)
    major = [p for p in rest if (p.get("share", 1.0) >= MIN_AUTO_SHARE or p.get("face")) and (p.get("face") or not any_face)]
    if len(major) != 1:
        return {"pid": None, "notice": ""}
    notice = []
    if not exps:
        notice.append("no_experimenter")
    if len(rest) > 1:
        notice.append("extra_people")
    return {"pid": major[0]["id"], "notice": " ".join(notice)}


def pick_child(participants: list[dict], experimenters: list[str], child_hint: str = "") -> int | None:
    """Return the participant id to crop, or None if the user has to choose."""
    return decide(participants, experimenters, child_hint)["pid"]


def experimenter_ids(participants: list[dict], experimenters: list[str]) -> set[int]:
    return {p["id"] for p in participants if is_experimenter(p, experimenters)}


def _nested(inner, outer, tol: int = 4) -> bool:
    """inner lies within outer and is clearly smaller."""
    if (inner[2] - inner[0]) * (inner[3] - inner[1]) >= 0.85 * (outer[2] - outer[0]) * (outer[3] - outer[1]):
        return False
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


def _fit_aspect(box, aspect=16 / 9):
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = w / aspect
    cy = (y0 + y1) / 2
    return [int(x0), int(round(cy - h / 2)), int(x1), int(round(cy - h / 2)) + int(round(h))]


def _sized_like(b, others) -> bool:
    """Same size as one of the other boxes: Zoom draws the tiles of a strip alike."""
    w, h = b[2] - b[0], b[3] - b[1]
    return any(abs((q[2] - q[0]) - w) <= 0.06 * w and abs((q[3] - q[1]) - h) <= 0.06 * h for q in others)


def build(analysis: dict, pid: int, overrides: list[dict] | None = None,
          exp_pids: set[int] | None = None) -> list[dict]:
    """exp_pids: participants known to be experimenters. Their windows are never the
    child's, and a frame where they are accounted for lets the child's window be
    taken by elimination when the child was not recognised in it."""
    H = analysis["video"]["height"]
    duration = analysis["video"]["duration"]
    tol = max(3, round(GROUP_TOL * H / 1080))
    rank = {"label": 0, "carry": 0, "anon": 0, "look": 1}
    exp_pids = set(exp_pids or ())

    # one entry per analysed frame: the child's tile, or why there is none. When the
    # same window was measured twice in a frame, prefer the box that is the size of the
    # other people's tiles (Zoom draws them alike), then the one that continues the
    # previous choice (no twitching between two measurements), then the better-backed.
    entries = []
    prev_box = None
    prev_t = -1e9
    for s in analysis["samples"]:
        mine = [t for t in s["tiles"] if t["pid"] == pid]
        peers = [t["box"] for t in s["tiles"] if t["pid"] is not None and t["pid"] != pid]
        exp_boxes = [t["box"] for t in s["tiles"] if t["pid"] in exp_pids]

        def overlaps_peer(b):  # tiles never overlap each other: a box that does is mis-measured
            return any(iou(b, q) >= 0.1 for q in peers)

        t = None
        flags = set()
        if mine:
            t = min(mine, key=lambda t: (rank.get(t["src"], 2), bool(t["clip"]), not _sized_like(t["box"], peers),
                                         overlaps_peer(t["box"]),
                                         not (prev_box is not None and iou(t["box"], prev_box) >= 0.5), -t["score"]))
            if t["src"] == "look":
                flags.add("look")
        else:
            # The child was not recognised in this frame. By elimination: live windows
            # that belong to nobody, once the people on screen are accounted for.
            # - a window continuing where the child's was a moment ago: the child, when the
            #   experimenter is seen elsewhere in the frame; otherwise tentatively ("held");
            # - else, with the experimenter seen, a single remaining window is the child's
            #   (a window the size of the experimenter's wins over slide scraps);
            # - a remaining window with nobody recognised could be anyone: left out, flagged.
            cands = [t for t in s["tiles"] if t["pid"] is None and t.get("live") and not overlaps_peer(t["box"])]
            recent = prev_box is not None and s["t"] - prev_t <= CONTINUE_GAP
            cont = [t for t in cands if recent and iou(t["box"], prev_box) >= 0.5]
            if cont:
                t = max(cont, key=lambda t: (iou(t["box"], prev_box), t["score"]))
                flags.add("inferred" if exp_boxes else "held")
            elif exp_boxes:
                pool = [t for t in cands if _sized_like(t["box"], exp_boxes)] or cands
                if len(pool) == 1:
                    t = pool[0]
                    flags.add("inferred")
                elif pool:
                    flags.add("unsure")
            elif cands:
                flags.add("unsure")
        if t is not None:
            box = t["box"]
            # a box that shrank into the previous one is a piece of the same window (an edge
            # inside the picture taken for the border): keep the window
            if prev_box is not None and t["src"] != "label" and _nested(box, prev_box):
                box = prev_box
            prev_box, prev_t = box, s["t"]
            if t["clip"]:
                flags.add("clipped")
            if t["flat"]:
                flags.add("camera_off")
            if t.get("cut") is not None:
                flags.add("clipped")
            entries.append((s["t"], box, flags, t.get("cut")))
        else:
            flags |= {"other"} if s["tiles"] else {"missing"}
            entries.append((s["t"], None, flags, None))

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
                same = bool(np.abs(ref - np.array(box)).max() <= tol) or iou([float(v) for v in ref], box) >= SAME_IOU
            # a moment the pipeline was unsure about must not stretch over the whole
            # span of the same box: cut the segment where the review marks change
            if same and (flags & SPLIT_FLAGS) != (last["flags"] & SPLIT_FLAGS):
                same = False
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
            (max(abs(a - b) for a, b in zip(prev["box"], nxt["box"])) <= 2 * tol or iou(prev["box"], nxt["box"]) >= SAME_IOU)
        if same_after and gap <= FILL_SAME:
            # the window was there before and after, at the same place: the detector
            # blinked, the output is continuous, nothing to look at
            s["box"] = list(prev["box"])
            s["flags"] = {"bridged"}
        elif nxt is not None and nxt["box"] is not None and gap <= FILL_ANY:
            s["box"] = list(prev["box"])
            s["flags"] = {"filled"}

    segs = _merge(_settle(segs, 2 * tol))
    for s in segs:
        if s["box"] is not None:
            s["box"] = _fit_aspect(s["box"])

    for ov in overrides or []:
        if ov.get("mode") != "ok":  # "ok" = looked at and left as is; handled after the review pass
            segs = _apply_override(segs, ov)

    out = []
    for s in segs:
        if s["end"] - s["start"] <= 1e-6:
            continue
        flags = sorted(s["flags"])
        out.append({"start": round(s["start"], 4), "end": round(s["end"], 4), "box": s["box"], "cut": s.get("cut"),
                    "status": "ok" if s["box"] is not None else "missing", "flags": flags, "review": False})
    # What a person should look at: spans without the child, and doubtful spans that add
    # up to at least MIN_REVIEW seconds in a row. A blink of a few frames where the box
    # was held or inferred makes no difference to anyone coding behaviour.
    i = 0
    while i < len(out):
        doubtful = out[i]["status"] == "missing" or any(f in REVIEW_FLAGS for f in out[i]["flags"])
        if not doubtful:
            i += 1
            continue
        j = i + 1
        while j < len(out) and (out[j]["status"] == "missing" or any(f in REVIEW_FLAGS for f in out[j]["flags"])) \
                and out[j]["start"] - out[j - 1]["end"] < 0.3:
            j += 1
        run = out[i:j]
        if any(s["status"] == "missing" for s in run) or sum(s["end"] - s["start"] for s in run) >= MIN_REVIEW:
            for s in run:
                s["review"] = True
        i = j
    for ov in overrides or []:  # spans the reviewer looked at and accepted
        if ov.get("mode") == "ok":
            for s in out:
                if s["review"] and s["start"] < ov["end"] - 1e-6 and s["end"] > ov["start"] + 1e-6:
                    s["review"] = False
                    s["flags"] = sorted(set(s["flags"]) | {"checked"})
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
                (max(abs(p - q) for p, q in zip(segs[j - 1]["box"], segs[j]["box"])) <= tol
                 or iou(segs[i]["box"], segs[j]["box"]) >= SAME_IOU):  # anchored to the run's start: no drifting chains
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
    review = [s for s in segments if s.get("review", s["status"] == "missing")]
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
