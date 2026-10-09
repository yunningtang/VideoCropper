"""Regression set: known tile boxes and the child's / experimenter's tile at given times.

    python tests/regress.py            # everything (analysis results are cached per code version)
    python tests/regress.py detect     # only the single-frame tile finder
    python tests/regress.py P124 P273  # only these videos

Truth lives in tests/truth.json: for each video, [time, box] pairs for the child's tile
and the experimenter's tile, checked by eye. Boxes count as found at IoU >= 0.8; the
child's timeline is right at a moment when its box has IoU >= 0.6 with the truth.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import av  # noqa: E402

from zoomcrop import timeline  # noqa: E402
from zoomcrop.detect import find_tiles, iou  # noqa: E402

TRUTH = json.load(open(ROOT / "tests" / "truth.json"))
CACHE = ROOT / "tests" / ".cache"


def frame_at(container, t):
    v = container.streams.video[0]
    container.seek(int(max(0, t - 0.01) / v.time_base), stream=v)
    for f in container.decode(v):
        if f.time >= t:
            return f


def check_detect(videos):
    """Single-frame tile finding at the truth moments: the geometric finder alone, and,
    when the learned model exists, the learned finder merged with it (what the
    analysis actually uses)."""
    from zoomcrop.tiledet import TileDetector, merge
    det = TileDetector() if TileDetector.available() else None
    tot = 0
    hit = {"geometric": 0, "merged": 0}
    ms = {"geometric": [], "merged": []}
    for vid in videos:
        truth = TRUTH["videos"][vid]
        cases = {}
        for role in ("child", "experimenter"):
            for t, box in truth[role]:
                cases.setdefault(t, []).append(box)
        with av.open(str(ROOT / TRUTH["files"][vid])) as c:
            for t, boxes in sorted(cases.items()):
                fr = frame_at(c, t)
                gray = fr.to_ndarray(format="gray")
                t0 = time.time()
                found = {"geometric": find_tiles(gray)}
                ms["geometric"].append((time.time() - t0) * 1000)
                if det is not None:
                    bgr = fr.to_ndarray(format="bgr24")
                    t0 = time.time()
                    found["merged"] = merge(det.detect(bgr, gray), find_tiles(gray))
                    ms["merged"].append((time.time() - t0) * 1000)
                for b in boxes:
                    tot += 1
                    for kind, tiles in found.items():
                        best = max((iou(b, tl.box) for tl in tiles), default=0.0)
                        hit[kind] += best >= 0.8
                        if best < 0.8:
                            print(f"  MISS {kind} {vid} t={t} {b} best {best:.2f}")
    for kind in found:
        print(f"detect ({kind}): {hit[kind]}/{tot} boxes found, {sum(ms[kind]) / len(ms[kind]):.0f} ms/frame mean, {max(ms[kind]):.0f} max")
    return hit["merged" if det is not None else "geometric"], tot


def code_hash():
    h = hashlib.sha1()
    for f in ("analyze.py", "detect.py", "tiledet.py", "models/tiles.onnx"):
        p = ROOT / "zoomcrop" / f
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:10]


def analysis(vid):
    from zoomcrop.analyze import Analyzer
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{vid}_{code_hash()}.json"
    if f.exists():
        return json.load(open(f))
    t0 = time.time()
    res = Analyzer().run(ROOT / TRUTH["files"][vid])
    res["took"] = round(time.time() - t0, 1)
    json.dump(res, open(f, "w"))
    return res


def box_at(segs, t):
    for s in segs:
        if s["start"] <= t < s["end"]:
            return s["box"]
    return None


def check_video(vid):
    truth = TRUTH["videos"][vid]
    res = analysis(vid)
    people = res["participants"]
    tl = {p["id"]: timeline.build(res, p["id"]) for p in people}

    def hits(pid, role):
        return sum(1 for t, b in truth[role] if (bx := box_at(tl[pid], t)) is not None and iou(bx, b) >= 0.6)

    scored = sorted(((hits(p["id"], "child"), hits(p["id"], "experimenter"), p) for p in people),
                    key=lambda x: (-x[0], x[1]))
    n_child, n_exp = len(truth["child"]), len(truth["experimenter"])
    child = scored[0] if scored else (0, 0, None)
    exp = max(((hits(p["id"], "experimenter"), p) for p in people if p is not child[2]), key=lambda x: x[0], default=(0, None))
    auto = timeline.decide(people, truth["experimenter_names"])
    name = child[2]["name"] if child[2] else "-"
    vis = timeline.summarize(tl[child[2]["id"]])["visible_pct"] if child[2] else 0
    print(f"{vid}: {len(people)} people; child={name!r} right {child[0]}/{n_child} visible {vis}%, "
          f"contaminated {child[1]}/{n_exp}; experimenter found {exp[0]}/{n_exp}"
          f"{' as ' + repr(exp[1]['name']) if exp[1] else ''}; auto-pick {auto['pid']} {auto['notice']!r}; "
          f"took {res.get('took', '?')}s, {res['stats']}")
    if truth["child_name"] and child[2] and child[2]["name"] != truth["child_name"]:
        print(f"  child should be named {truth['child_name']!r}")
    for t, b in truth["child"]:
        bx = box_at(tl[child[2]["id"]], t) if child[2] else None
        if bx is None or iou(bx, b) < 0.6:
            print(f"  child wrong at t={t}: truth {b}, got {bx}")
    # The same with the experimenter known (as it is once they are remembered by name or
    # face): the child's window is then also taken by elimination where the child was
    # not recognised in it. This must gain visibility without picking up the experimenter.
    if child[2] and exp[1]:
        ex_pid = exp[1]["id"]
        segs = timeline.build(res, child[2]["id"], None, {ex_pid})
        sm = timeline.summarize(segs)
        right = sum(1 for t, b in truth["child"] if (bx := box_at(segs, t)) is not None and iou(bx, b) >= 0.6)
        bad = sum(1 for t, b in truth["experimenter"] if (bx := box_at(segs, t)) is not None and iou(bx, b) >= 0.6)
        held = sum(s["end"] - s["start"] for s in segs if "held" in s["flags"])
        inferred = sum(s["end"] - s["start"] for s in segs if "inferred" in s["flags"])
        print(f"  with experimenter known: right {right}/{n_child} visible {sm['visible_pct']}%, contaminated {bad}/{n_exp}, "
              f"to check {sm['review_n']} spans {sm['review_s']}s; inferred {inferred:.1f}s, held {held:.1f}s; "
              f"auto-pick {timeline.decide([dict(p, experimenter=p['id'] == ex_pid) for p in people], [])}")
        for t, b in truth["child"]:
            bx = box_at(segs, t)
            if bx is None or iou(bx, b) < 0.6:
                print(f"    child wrong at t={t}: truth {b}, got {bx}")
        for t, b in truth["experimenter"]:
            bx = box_at(segs, t)
            if bx is not None and iou(bx, b) >= 0.6:
                print(f"    experimenter cropped at t={t}: {bx}")
        return right, n_child, bad
    return child[0], n_child, child[1]


if __name__ == "__main__":
    args = sys.argv[1:]
    videos = [a for a in args if a in TRUTH["videos"]] or list(TRUTH["videos"])
    if "detect" in args or not args or all(a in TRUTH["videos"] for a in args):
        check_detect(videos)
    if "detect" not in args:
        tot = ok = bad = 0
        for vid in videos:
            if vid == "P246":
                continue  # boxes only
            h, n, c = check_video(vid)
            ok += h
            tot += n
            bad += c
        print(f"child timeline: {ok}/{tot} moments right, {bad} contaminated by the experimenter")
