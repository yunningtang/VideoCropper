"""Local web app: drop videos in, watch progress, review and correct the crops.

Everything runs on this machine; the server only listens on 127.0.0.1.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

import numpy as np

from . import timeline, work
from .analyze import FACE_MIN_VOTES, FACE_SAME, Cancelled, FaceReader, code_version, names_match

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "workspace"
UPLOADS = WORK / "uploads"
JOBDIR = WORK / "jobs"
SETTINGS_FILE = WORK / "settings.json"
STATIC = Path(__file__).resolve().parent / "static"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".wmv"}
N_WORKERS = 4        # videos analysed or rendered at the same time (each in its own process)
MAX_ATTEMPTS = 2     # a task that fails is run once more before the video is marked as failed
CACHE = WORK / "cache"  # analysis results by video content, so a re-added video is not analysed again

# known_faces: experimenters remembered by face, [{"name", "model", "vectors": [[...], ...]}].
# Only face descriptions of people the user marked as experimenters are ever stored.
DEFAULT_SETTINGS = {"experimenters": [], "known_faces": [], "out_height": 360, "export_dir": ""}
FACES_PER_PERSON = 24

# Runs in its own process: Tk wants a main thread, and a crash there must not take the server down.
FOLDER_DIALOG = """
import sys, tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
root.update()
path = filedialog.askdirectory(parent=root, initialdir=sys.argv[1] or None,
                               title="Choose where to save the exported files", mustexist=False)
sys.stdout.buffer.write(path.encode("utf-8"))
"""

app = FastAPI(title="Zoom child crop")
lock = threading.RLock()


@app.middleware("http")
async def same_origin_only(request: Request, call_next):
    """Browsers send Origin on cross-site requests; a page from the internet must not be
    able to add videos, change settings or open folder dialogs on this computer."""
    origin = request.headers.get("origin")
    if origin and request.method in ("POST", "PUT", "DELETE"):
        host = origin.split("//", 1)[-1].split(":", 1)[0]
        if host not in ("127.0.0.1", "localhost", "[::1]"):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=403, content={"detail": "Requests from other sites are not accepted."})
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    """A bug in a request handler: keep the traceback (workspace/server.errors.log) and
    tell the page what went wrong instead of a bare 500."""
    from fastapi.responses import JSONResponse
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        with open(WORK / "server.errors.log", "a", encoding="utf-8") as f:
            f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} {request.method} {request.url.path}\n"
                    f"{''.join(traceback.format_exception(exc))}\n")
    except OSError:
        pass
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})
jobs: dict[str, dict] = {}
cancels: dict[str, threading.Event] = {}
class TaskQueue(queue.PriorityQueue):
    """Renders (a fix's preview file, a chosen child's first output) go before analyses,
    so a manual fix does not wait behind a batch of long analyses; otherwise first in, first out."""
    _n = 0

    def put(self, item, block=True, timeout=None):
        jid, task = item
        TaskQueue._n += 1
        super().put((0 if task in ("render", "rerender") else 1, TaskQueue._n, jid, task), block, timeout)

    def get(self, block=True, timeout=None):
        _, _, jid, task = super().get(block, timeout)
        return jid, task


tasks: queue.Queue = TaskQueue()
settings: dict = dict(DEFAULT_SETTINGS)
# Face descriptions of the people in each analysed video, by participant id. They sit in
# the job's folder (faces.json) until the child is chosen, so that an experimenter can
# still be remembered by face after a restart; the child's are deleted at that moment.
job_faces: dict[str, dict[int, list]] = {}
FACES_FILE = "faces.json"


# -- persistence -----------------------------------------------------------
def load_state():
    for d in (UPLOADS, JOBDIR):
        d.mkdir(parents=True, exist_ok=True)
    if SETTINGS_FILE.exists():
        try:
            settings.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    # earlier versions wrote results straight into a fixed folder; offer it as the first export location
    settings.setdefault("known_faces", [])
    old = settings.pop("output_dir", None)
    if old and not settings.get("export_dir"):
        settings["export_dir"] = old
    for d in (JOBDIR.iterdir() if JOBDIR.exists() else []):
        if d.is_dir() and not (d / "job.json").exists():
            shutil.rmtree(d, ignore_errors=True)  # removed earlier while a file in it was still open
    for f in sorted(JOBDIR.glob("*/job.json")):
        try:
            job = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        jobs[job["id"]] = job
        cancels[job["id"]] = threading.Event()
        # resume whatever was interrupted by the last shutdown
        job["rendering"] = None
        job["rerender_queued"] = False
        if job["status"] in ("queued", "analyzing"):
            enqueue(job, "analyze")
        elif job["status"] == "rendering":
            enqueue(job, "render")
        elif job["status"] == "done" and job.get("tl_version") != timeline.VERSION and job.get("child_pid") is not None                 and (JOBDIR / job["id"] / "analysis.json").exists():
            # the timeline rules changed since this video was finished: rebuild its segments from
            # the analysis (seconds) and refresh the preview file; nothing is analysed again
            try:
                analysis = read_json(JOBDIR / job["id"] / "analysis.json")
                segs = timeline.build(analysis, job["child_pid"], job["overrides"], other_people(job))
                (JOBDIR / job["id"] / "segments.json").write_text(json.dumps(segs), encoding="utf-8")
                job["summary"] = timeline.summarize(segs)
                job["tl_version"] = timeline.VERSION
                job["rev"] += 1
                job["rerender_queued"] = True
                tasks.put((job["id"], "rerender"))
                save_job(job)
            except Exception:
                traceback.print_exc()
        elif job["status"] == "done" and job.get("output_rev", job["rev"]) != job["rev"]:
            job["rerender_queued"] = True  # a fix was saved but its preview file never got made
            tasks.put((job["id"], "rerender"))


def save_job(job: dict):
    if job["id"] not in jobs:
        return  # removed meanwhile: a worker finishing late must not bring the folder back
    d = JOBDIR / job["id"]
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, d / "job.json")


def public_settings() -> dict:
    """Settings as the page sees them: the face descriptions stay on the server."""
    out = dict(settings)
    out["known_faces"] = [{"name": e["name"], "n": len(e.get("vectors", []))} for e in settings["known_faces"]]
    return out


def save_settings():
    WORK.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=1), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# -- jobs ------------------------------------------------------------------
def new_job(src: Path, origin: str, batch: str = "") -> dict:
    job = {
        "batch": batch,
        "id": uuid.uuid4().hex[:10], "name": src.name if origin == "path" else src.name.split("_", 1)[1],
        "label": "", "src": str(src), "origin": origin, "status": "queued", "progress": 0.0, "error": "",
        "created": time.time(), "child_pid": None, "child_name": "", "participants": [],
        "video": None, "summary": None, "output": "", "overrides": [], "rev": 0, "took": {},
        "reviewed": False, "started": None, "exported": None, "render": None, "notice": "",
    }
    with lock:
        jobs[job["id"]] = job
        cancels[job["id"]] = threading.Event()
        save_job(job)
    enqueue(job, "analyze")
    return job


def enqueue(job: dict, task: str, attempt: int = 0):
    job["status"] = "queued"
    job["progress"] = 0.0
    job["next"] = task
    job["attempt"] = attempt
    tasks.put((job["id"], task))


# -- analysis cache --------------------------------------------------------
def video_key(src: str) -> str:
    """Fingerprint of a video's content (size, first and last megabyte), the analysis
    code and the remembered faces: the same video gives the same analysis as long as
    none of these changed."""
    import hashlib
    p = Path(src)
    n = p.stat().st_size
    h = hashlib.sha1(str(n).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        if n > 2 << 20:
            f.seek(-(1 << 20), 2)
            h.update(f.read(1 << 20))
    h.update(json.dumps(settings.get("known_faces") or [], sort_keys=True).encode())
    return f"{h.hexdigest()[:16]}-{code_version()}"


def cache_get(key: str, jdir: Path) -> dict | None:
    """Copy a cached analysis (result and participant pictures) into the job's folder."""
    d = CACHE / key
    if not (d / "analysis.json").exists():
        return None
    jdir.mkdir(parents=True, exist_ok=True)
    for f in d.iterdir():
        shutil.copyfile(f, jdir / f.name)
    res = read_json(jdir / "analysis.json")
    # face descriptions are never cached (the child's would outlive the job), so a
    # cached video cannot teach a new face; Re-analyze runs a fresh analysis
    return {"participants": res["participants"], "video": res["video"], "faces": {}}


def cache_put(key: str, jdir: Path):
    d = CACHE / key
    d.mkdir(parents=True, exist_ok=True)
    for f in jdir.iterdir():
        if f.name == "analysis.json" or (f.suffix == ".jpg" and f.name.startswith("p")):
            shutil.copyfile(f, d / f.name)


def set_status(job: dict, status: str, **extra):
    with lock:
        job["status"] = status
        job.update(extra)
        save_job(job)


def faces_of(jid: str) -> dict[int, list]:
    if jid not in job_faces:
        out: dict[int, list] = {}
        try:
            data = read_json(JOBDIR / jid / FACES_FILE)
            if data.get("model") == FaceReader.MODEL:
                out = {int(k): v for k, v in data.get("faces", {}).items()}
        except (OSError, ValueError, AttributeError):
            pass
        job_faces[jid] = out
    return job_faces[jid]


def save_faces(jid: str, faces: dict[int, list]):
    job_faces[jid] = faces
    f = JOBDIR / jid / FACES_FILE
    if faces:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"model": FaceReader.MODEL, "faces": {str(k): v for k, v in faces.items()}}),
                     encoding="utf-8")
    else:
        f.unlink(missing_ok=True)


def drop_child_faces(job: dict):
    """The person being cropped is a child: nothing about their face is kept."""
    faces = faces_of(job["id"])
    if job.get("child_pid") in faces:
        save_faces(job["id"], {k: v for k, v in faces.items() if k != job["child_pid"]})


def link_named_faces(job: dict) -> bool:
    """An experimenter known by name whose face was seen: remember the face under that
    name, so that recordings without name labels recognise them too."""
    faces = faces_of(job["id"])
    changed = False
    for p in job["participants"]:
        if p.get("anon") or p["id"] == job.get("child_pid") or len(faces.get(p["id"], [])) < FACE_MIN_VOTES:
            continue
        if any(names_match(p["name"], e) for e in settings["experimenters"]):
            remember_face(p["name"], faces[p["id"]])
            changed = True
    return changed


def other_people(job: dict) -> set[int]:
    """Everyone but the chosen child who is a real person (showed a face, or is a
    registered experimenter): their windows can never be the child's."""
    return {p["id"] for p in job["participants"] if p["id"] != job.get("child_pid")
            and (p.get("face") or timeline.is_experimenter(p, settings["experimenters"]))}


def child_like_experimenter(job: dict) -> bool:
    """Does the face of the person about to be cropped match a remembered experimenter?
    A child mistaken for an experimenter would otherwise be excluded silently and the
    experimenter cropped instead; this is checked before the child's vectors are dropped."""
    vectors = faces_of(job["id"]).get(job.get("child_pid")) or []
    known = [e for e in settings["known_faces"] if e.get("model") == FaceReader.MODEL and e.get("vectors")]
    if not vectors or not known:
        return False
    votes = sum(1 for v in vectors if max(max(_cos(v, k) for k in e["vectors"]) for e in known) >= FACE_SAME)
    return votes >= FACE_MIN_VOTES


def settle(job: dict, queue_render: bool = True) -> bool:
    """Decide who to crop without asking, if the rule allows it; queue the render then."""
    d = timeline.decide(job["participants"], settings["experimenters"])
    job["notice"] = d["notice"]
    if d["pid"] is None:
        return False
    job["child_pid"] = d["pid"]
    if child_like_experimenter(job):
        job["notice"] = (job["notice"] + " child_like_experimenter").strip()
    if link_named_faces(job):
        save_settings()
    drop_child_faces(job)
    if queue_render:
        enqueue(job, "render")
    return True


def _cos(a, b) -> float:
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d > 1e-6 else 0.0


def apply_known_faces(job: dict) -> bool:
    """Mark participants whose face matches a remembered experimenter. Used when the
    list of remembered faces changes after a video was analysed."""
    vectors = faces_of(job["id"])
    changed = False
    for p in job["participants"]:
        if p.get("experimenter") or not p.get("anon") or p["id"] not in vectors:
            continue
        votes: dict[str, int] = {}
        for v in vectors[p["id"]]:
            best = max(((max(_cos(v, k) for k in e["vectors"]), e["name"]) for e in settings["known_faces"]
                        if e.get("model") == FaceReader.MODEL and e.get("vectors")), default=(0.0, ""))
            if best[0] >= FACE_SAME:
                votes[best[1]] = votes.get(best[1], 0) + 1
        if votes:
            name, n = max(votes.items(), key=lambda kv: kv[1])
            if n >= FACE_MIN_VOTES:
                p["experimenter"], p["name"], changed = True, name, True
    return changed


def remember_face(name: str, vectors: list):
    """Add face descriptions to a remembered experimenter (or create one)."""
    name = name.strip()
    if not name or not vectors:
        return
    entry = next((e for e in settings["known_faces"] if names_match(e["name"], name) and e.get("model") == FaceReader.MODEL), None)
    if entry is None:
        entry = {"name": name, "model": FaceReader.MODEL, "vectors": []}
        settings["known_faces"].append(entry)
    entry["vectors"] = (entry["vectors"] + [[round(float(x), 5) for x in v] for v in vectors])[-FACES_PER_PERSON:]


def worker():
    """Takes tasks off the queue and runs each in a child process (see work.py)."""
    while True:
        jid, task = tasks.get()
        job = jobs.get(jid)
        cancel = cancels.get(jid)
        if job is None or cancel is None or cancel.is_set():
            continue
        jdir = JOBDIR / jid

        def progress(p, job=job):
            job["progress"] = round(float(p), 4)

        try:
            if task == "analyze":
                set_status(job, "analyzing", progress=0.0, error="", started=time.time())
                t0 = time.time()
                jdir.mkdir(parents=True, exist_ok=True)
                key = video_key(job["src"])
                res = None if job.pop("fresh", False) else cache_get(key, jdir)
                job["cached"] = res is not None
                if res is None:
                    res = work.run(work.analyze_task, (job["src"], str(jdir), settings.get("known_faces") or []),
                                   progress, cancel.is_set)
                    if cancel.is_set():
                        continue  # removed while the analysis was finishing
                    cache_put(key, jdir)
                save_faces(jid, res["faces"])
                job["took"]["analyze"] = round(time.time() - t0, 1)
                job["participants"] = res["participants"]
                job["video"] = res["video"]
                with lock:
                    settled = settle(job, queue_render=False)
                if not settled:
                    set_status(job, "choose", progress=1.0)
                    continue
                task = "render"
            if task == "render":
                analysis = read_json(jdir / "analysis.json")
                pid = job["child_pid"]
                job["child_name"] = next((p["name"] for p in job["participants"] if p["id"] == pid), "")
                segs = timeline.build(analysis, pid, job["overrides"], other_people(job))
                set_status(job, "rendering", progress=0.0, error="", started=time.time())
                t0 = time.time()
                # The result stays in the workspace for review; nothing is written
                # anywhere else until the user exports it. Each render gets its own file
                # name: the previous one may still be open in the browser's player, and
                # Windows will not let an open file be replaced.
                out = jdir / f"child_r{job['rev'] + 1}.mp4"
                info = work.run(work.render_task, (job["src"], str(out), segs, int(settings["out_height"])),
                                progress, cancel.is_set)
                if cancel.is_set():
                    continue
                job["took"]["render"] = round(time.time() - t0, 1)
                for old in jdir.glob("child*.mp4"):
                    if old != out:
                        try:
                            old.unlink()
                        except OSError:
                            pass  # still being played; removed with the job, or by the next render
                (jdir / "segments.json").write_text(json.dumps(segs), encoding="utf-8")
                job["tl_version"] = timeline.VERSION
                set_status(job, "done", progress=1.0, summary=timeline.summarize(segs), output=str(out),
                           render=info, rev=job["rev"] + 1, output_rev=job["rev"] + 1, reviewed=False)
            if task == "rerender":
                # A manual fix was applied: the timeline was rebuilt on the spot (set_overrides)
                # and the review screen shows it from the source video; only the preview
                # file is refreshed here, in the background, without changing the status.
                if job.get("rendering"):
                    tasks.put((jid, task))  # one render per video at a time
                    time.sleep(1)
                    continue
                job["rerender_queued"] = False
                segs = read_json(jdir / "segments.json")
                rev = job["rev"]
                job["rendering"] = {"progress": 0.0}

                def rprogress(p, job=job):
                    job["rendering"] = {"progress": round(float(p), 4)}

                t0 = time.time()
                out = jdir / f"child_r{rev}.mp4"
                try:
                    info = work.run(work.render_task, (job["src"], str(out), segs, int(settings["out_height"])),
                                    rprogress, cancel.is_set)
                finally:
                    job["rendering"] = None
                for old in jdir.glob("child*.mp4"):
                    if old != out:
                        try:
                            old.unlink()
                        except OSError:
                            pass
                with lock:
                    job["took"]["render"] = round(time.time() - t0, 1)
                    job.update(output=str(out), render=info, output_rev=rev)
                    if job["rev"] != rev and not job.get("rerender_queued"):  # fixed again meanwhile
                        job["rerender_queued"] = True
                        tasks.put((jid, "rerender"))
                    save_job(job)
        except Cancelled:
            pass
        except Exception as e:  # keep the worker alive; show the failure on the job
            if cancel.is_set() or jid not in jobs:
                continue  # removed meanwhile: nothing to report, nothing to write
            jdir.mkdir(parents=True, exist_ok=True)
            with open(jdir / "error.log", "a", encoding="utf-8") as f:
                f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} {task}, attempt {job.get('attempt', 0) + 1}\n{traceback.format_exc()}\n")
            if task == "rerender":
                # the reviewed result stays usable; the stale preview file is reported, not fatal
                with lock:
                    job["rendering"] = None
                    job["render_error"] = f"{type(e).__name__}: {e}".splitlines()[-1][:200]
                    save_job(job)
                continue
            if job.get("attempt", 0) + 1 < MAX_ATTEMPTS:
                # a decoder hiccup or a file briefly locked by another program: once more
                with lock:
                    enqueue(job, task, attempt=job.get("attempt", 0) + 1)
                    save_job(job)
            else:
                set_status(job, "error", error=f"{type(e).__name__}: {e}")


def public(job: dict, full: bool = False) -> dict:
    """What the page sees. The list polled every second carries the participants only
    for a video waiting for a choice (the picker needs them); the job detail has
    everything, so that 400 queued videos do not make every poll megabytes."""
    keys = ("id", "name", "origin", "status", "progress", "error", "child_pid", "child_name",
            "video", "summary", "rev", "took", "created", "src",
            "reviewed", "started", "exported", "notice", "cached", "attempt", "rendering", "output_rev", "render_error", "label", "batch")
    out = {k: job.get(k) for k in keys}
    out["n_overrides"] = len(job.get("overrides") or [])
    if full or job["status"] == "choose":
        out["participants"] = job.get("participants")
    return out


def get_job(jid: str) -> dict:
    job = jobs.get(jid)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


# -- routes ----------------------------------------------------------------
@app.on_event("startup")
def startup():
    load_state()
    for _ in range(N_WORKERS):
        threading.Thread(target=worker, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store"})


@app.get("/api/state")
def state():
    with lock:
        return {"jobs": [public(j) for j in sorted(jobs.values(), key=lambda j: j["created"])],
                "settings": public_settings(), "now": time.time()}


@app.put("/api/upload")
async def upload(request: Request, name: str, batch: str = ""):
    safe = Path(name).name
    batch = "".join(ch for ch in Path(batch).name.strip() if ch not in '\\/:*?"<>|')[:40]
    if Path(safe).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, f"Unsupported file type: {safe}")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dst = UPLOADS / f"{uuid.uuid4().hex[:6]}_{safe}"
    part = dst.with_name(dst.name + ".part")
    try:
        with open(part, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        os.replace(part, dst)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return public(new_job(dst, "upload", batch))


@app.post("/api/add_path")
async def add_path(request: Request):
    body = await request.json()
    p = Path(str(body.get("path", "")).strip().strip('"'))
    if not p.exists():
        raise HTTPException(400, f"Path does not exist: {p}")
    files = [p] if p.is_file() else sorted(f for f in p.iterdir() if f.is_file())
    files = [f for f in files if f.suffix.lower() in VIDEO_EXT and not f.stem.endswith("_child")]
    if not files:
        raise HTTPException(400, "No video files found at that path")
    known = {j["src"] for j in jobs.values()}
    # videos added from a folder keep its name as their batch (B01, B02, ...)
    batch = p.name if p.is_dir() else ""
    added = [public(new_job(f, "path", batch)) for f in files if str(f) not in known]
    return {"added": added, "skipped": len(files) - len(added)}


@app.get("/api/jobs/{jid}")
def job_detail(jid: str):
    job = get_job(jid)
    out = public(job, full=True)
    seg_file = JOBDIR / jid / "segments.json"
    out["segments"] = read_json(seg_file) if seg_file.exists() else []
    out["overrides"] = job["overrides"]
    out["src_exists"] = Path(job["src"]).exists()
    out["flag_text"] = timeline.FLAG_TEXT
    out["notice_text"] = " ".join(timeline.NOTICE_TEXT.get(k, k) for k in (job.get("notice") or "").split())
    return out


@app.post("/api/jobs/{jid}/child")
async def choose_child(jid: str, request: Request):
    job = get_job(jid)
    body = await request.json()
    pid = int(body["pid"])
    if pid != -1 and not any(p["id"] == pid for p in job["participants"]):
        raise HTTPException(400, "unknown participant")
    if not (JOBDIR / jid / "analysis.json").exists():
        raise HTTPException(409, "analysis not finished")
    if job["status"] not in ("choose", "done", "error") or job.get("rendering") or job.get("rerender_queued"):
        raise HTTPException(409, "This video is still being processed; try again in a moment.")
    with lock:
        job["child_pid"] = pid
        job["notice"] = ""  # the user decided
        # Only names the page listed explicitly are remembered as experimenters, and never
        # the person being cropped: picking someone must not silently relabel anyone.
        chosen = next((p["name"] for p in job["participants"] if p["id"] == pid), "")
        for name in body.get("experimenters", []):
            if any(p["name"] == name and not p.get("anon") for p in job["participants"]) and name != chosen \
                    and not any(names_match(name, e) for e in settings["experimenters"]):
                settings["experimenters"].append(name)
        # people without a name label can be remembered by face instead; never the child
        faces = faces_of(jid)
        for item in body.get("remember_faces", []):
            fpid = int(item.get("pid", -1))
            vectors = faces.get(fpid)
            if fpid == pid or not any(p["id"] == fpid for p in job["participants"]):
                continue
            if not vectors:
                raise HTTPException(409, "The face descriptions for this video are no longer available. "
                                         "Re-analyze it to remember this person by face.")
            remember_face(str(item.get("name", "")), vectors)
            for p in job["participants"]:
                if p["id"] == fpid:
                    p["experimenter"], p["name"] = True, str(item.get("name", "")).strip() or p["name"]
        link_named_faces(job)  # a named experimenter's face goes with the name
        if child_like_experimenter(job):
            job["notice"] = "child_like_experimenter"
        drop_child_faces(job)
        save_settings()
        enqueue(job, "render")
        save_job(job)
        for other in jobs.values():
            if other["status"] == "choose" and other is not job:
                apply_known_faces(other)
                settle(other)
                save_job(other)
    return public(job)


@app.post("/api/jobs/{jid}/overrides")
async def set_overrides(jid: str, request: Request):
    job = get_job(jid)
    body = await request.json()
    if job["status"] not in ("done", "error") or job["child_pid"] is None:
        raise HTTPException(409, "job is busy")
    import math
    dur = float((job.get("video") or {}).get("duration") or 0) or None
    W, H = (job.get("video") or {}).get("width"), (job.get("video") or {}).get("height")
    clean = []
    for ov in body.get("overrides", []):
        try:
            a, b = float(ov["start"]), float(ov["end"])
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "A fix needs a start and an end time.")
        if not (math.isfinite(a) and math.isfinite(b)):
            raise HTTPException(400, "A fix needs finite start and end times.")
        a = max(0.0, a)
        if dur:
            b = min(dur, b)
        if b <= a:
            continue  # entirely outside the video, or empty
        item = {"start": round(a, 4), "end": round(b, 4)}
        if ov.get("mode") in ("hold", "ok"):
            item["mode"] = ov["mode"]
        elif ov.get("box"):
            try:
                box = [int(round(float(v))) for v in ov["box"]]
            except (TypeError, ValueError):
                raise HTTPException(400, "A box needs four numbers.")
            if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] - box[0] < 8 or box[3] - box[1] < 8:
                raise HTTPException(400, "A box needs a left, top, right and bottom edge, at least 8 px apart.")
            if W and H and (min(W, box[2]) - max(0, box[0]) < 8 or min(H, box[3]) - max(0, box[1]) < 8):
                raise HTTPException(400, f"That box ({box[0]},{box[1]} to {box[2]},{box[3]}) lies outside the picture ({W}×{H}).")
            item["box"] = box
        else:
            item["box"] = None
        clean.append(item)
    jdir = JOBDIR / jid
    analysis = await asyncio.to_thread(read_json, jdir / "analysis.json")
    segs = timeline.build(analysis, job["child_pid"], clean, other_people(job))
    with lock:
        job["overrides"] = clean
        job["rev"] += 1
        job["summary"] = timeline.summarize(segs)
        job["reviewed"] = False
        (jdir / "segments.json").write_text(json.dumps(segs), encoding="utf-8")
        job["tl_version"] = timeline.VERSION
        if not job.get("rerender_queued"):  # fixes arrive every few seconds; one render follows them
            job["rerender_queued"] = True
            tasks.put((jid, "rerender"))
        save_job(job)
    out = public(job)
    out["segments"] = segs
    return out


@app.post("/api/jobs/{jid}/label")
async def set_label(jid: str, request: Request):
    """The participant id used to name the exported files (defaults to the file's stem)."""
    job = get_job(jid)
    body = await request.json()
    label = "".join(ch for ch in str(body.get("label", "")).strip() if ch not in '\\/:*?"<>|')[:60]
    with lock:
        job["label"] = label
        save_job(job)
    return public(job)


@app.post("/api/jobs/{jid}/clone")
async def clone_job(jid: str, request: Request):
    """A second person to crop out of the same recording (two participants in one Zoom
    session): a new job that reuses this one's analysis and asks who to crop."""
    job = get_job(jid)
    body = await request.json()
    jdir = JOBDIR / jid
    if job["status"] not in ("done", "choose") or not (jdir / "analysis.json").exists():
        raise HTTPException(409, "Finish this video's analysis first.")
    new = {
        "id": uuid.uuid4().hex[:10], "name": job["name"], "label": str(body.get("label", "")).strip()[:60], "batch": job.get("batch", ""),
        "src": job["src"], "origin": "clone", "status": "choose", "progress": 1.0, "error": "",
        "created": time.time(), "child_pid": None, "child_name": "", "participants": json.loads(json.dumps(job["participants"])),
        "video": job["video"], "summary": None, "output": "", "overrides": [], "rev": 0, "took": dict(job.get("took") or {}),
        "reviewed": False, "started": None, "exported": None, "render": None, "notice": "", "cached": True,
    }
    ndir = JOBDIR / new["id"]
    ndir.mkdir(parents=True, exist_ok=True)
    for f in jdir.iterdir():
        if f.name == "analysis.json" or (f.suffix == ".jpg" and f.name.startswith("p")) or f.name == "strip.jpg":
            shutil.copyfile(f, ndir / f.name)
    with lock:
        jobs[new["id"]] = new
        cancels[new["id"]] = threading.Event()
        save_job(new)
    return public(new, full=True)


@app.post("/api/jobs/{jid}/reviewed")
async def set_reviewed(jid: str, request: Request):
    job = get_job(jid)
    body = await request.json()
    with lock:
        job["reviewed"] = bool(body.get("reviewed"))
        save_job(job)
    if job["reviewed"]:  # the playback copy is only needed while the video is being reviewed
        try:
            (JOBDIR / jid / "source.mp4").unlink(missing_ok=True)
        except OSError:
            pass
    return public(job)


def summary_text(rows: list[tuple[dict, str]]) -> str:
    """One row per video, for keeping track of a batch (opens in Excel)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["batch", "participant", "video", "cropped_person", "duration_s", "visible_pct", "missing_s", "review_spans",
                "review_s", "window_changes", "manual_fixes", "reviewed", "exported_file"])
    for j, fname in rows:
        sm = j.get("summary") or {}
        w.writerow([j.get("batch", ""), j.get("label") or Path(j["name"]).stem, j["name"], j.get("child_name", ""), sm.get("duration", ""), sm.get("visible_pct", ""),
                    sm.get("missing_s", ""), sm.get("review_n", ""), sm.get("review_s", ""),
                    sm.get("layout_changes", ""), len(j.get("overrides") or []),
                    "yes" if j.get("reviewed") else "no", fname])
    return "\ufeff" + buf.getvalue()  # BOM so Excel reads it as UTF-8


@app.post("/api/pick_folder")
async def pick_folder(request: Request):
    """Show the system's folder dialog on this computer and return the chosen path."""
    body = await request.json()
    start = str(body.get("start") or "")
    if not Path(start).is_dir():
        start = ""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", FOLDER_DIALOG, start],
                                    capture_output=True, timeout=900, creationflags=flags)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise HTTPException(500, f"Could not open the folder dialog ({e}). Type the path instead.")
    if r.returncode != 0:
        raise HTTPException(500, "Could not open the folder dialog. Type the path instead.")
    path = r.stdout.decode("utf-8", "replace").strip()
    return {"path": str(Path(path)) if path else ""}


@app.post("/api/export")
async def export(request: Request):
    """Copy finished results to a folder the user chose."""
    body = await request.json()
    raw = str(body.get("dest") or "").strip().strip('"')
    if not raw or not Path(raw).is_absolute():
        raise HTTPException(400, "Choose a folder to export to (a full path such as D:\\Study\\cropped).")
    dest = Path(raw)
    want = body.get("include") or {}
    with lock:
        chosen = [jobs[i] for i in body.get("ids", []) if i in jobs and jobs[i]["status"] == "done"
                  and jobs[i].get("output") and Path(jobs[i]["output"]).exists()]
    if not chosen:
        raise HTTPException(400, "None of the selected videos has a finished result to export.")
    behind = [j["name"] for j in chosen if j.get("rendering") or j.get("output_rev", j["rev"]) != j["rev"]]
    if behind:
        raise HTTPException(409, "The preview file is still being re-exported after a manual fix for: "
                                 f"{', '.join(behind[:4])}{'…' if len(behind) > 4 else ''}. Try again in a minute.")
    if not any(want.get(k) for k in ("video", "segments", "report", "summary")):
        raise HTTPException(400, "Tick at least one kind of file to save.")

    # plan first, so that nothing is written if the user still has to confirm replacing files
    plan: list[tuple[dict, str, Path]] = []
    stems: dict[str, int] = {}
    rows = []
    by_batch = bool(body.get("by_batch"))  # B01/, B02/, ... under the chosen folder
    for j in sorted(chosen, key=lambda j: (j.get("batch", ""), j.get("label") or j["name"])):
        stem = j.get("label") or Path(j["name"]).stem
        stems[stem] = stems.get(stem, 0) + 1
        if stems[stem] > 1:  # two sources with the same file name
            stem = f"{stem}_{stems[stem]}"
        folder = dest / j["batch"] if by_batch and j.get("batch") else dest
        rows.append((j, f"{stem}_child.mp4"))
        if want.get("video"):
            plan.append((j, "video", folder / f"{stem}_child.mp4"))
        if want.get("segments"):
            plan.append((j, "segments", folder / f"{stem}_segments.csv"))
        if want.get("report"):
            plan.append((j, "report", folder / f"{stem}_report.json"))
    targets = [t for _, _, t in plan] + ([dest / "crop_summary.csv"] if want.get("summary") else [])
    existing = [str(t.relative_to(dest)) for t in targets if t.exists()]
    if existing and not body.get("overwrite"):
        return {"conflicts": existing[:8], "n_conflicts": len(existing)}

    def run():
        dest.mkdir(parents=True, exist_ok=True)
        for j, kind, target in plan:
            target.parent.mkdir(parents=True, exist_ok=True)
            segs = read_json(JOBDIR / j["id"] / "segments.json")
            if kind == "video":
                # results made by older versions already sit in their folder under this name
                if Path(j["output"]).resolve() != target.resolve():
                    shutil.copyfile(j["output"], target)
            elif kind == "segments":
                timeline.write_csv(segs, target)
            else:
                report = {
                    "source": j["src"], "video": j["video"], "render": j.get("render"),
                    "cropped_person": {"pid": j["child_pid"], "name": j.get("child_name", "")},
                    "participants": j["participants"], "summary": j["summary"], "reviewed": bool(j.get("reviewed")),
                    "manual_overrides": j["overrides"], "segments": segs, "flag_meanings": timeline.FLAG_TEXT,
                    "exported": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                target.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        if want.get("summary"):
            # newline="": the csv module already wrote \r\n line ends
            with open(dest / "crop_summary.csv", "w", encoding="utf-8", newline="") as f:
                f.write(summary_text(rows))

    try:
        await asyncio.to_thread(run)
    except OSError as e:
        raise HTTPException(400, f"Could not write to {dest}: {e.strerror or e}")
    with lock:
        for j in chosen:
            if j["id"] not in jobs:
                continue
            j["exported"] = {"dir": str(dest), "time": time.time(), "rev": j["rev"]}
            save_job(j)
        settings["export_dir"] = str(dest)
        save_settings()
    return {"videos": len(chosen), "files": len(targets), "dir": str(dest)}


@app.post("/api/open_folder")
async def open_folder(request: Request):
    """Open a folder in the file manager. Only folders this app exported to."""
    body = await request.json()
    p = Path(str(body.get("path") or ""))
    known = {settings.get("export_dir")} | {(j.get("exported") or {}).get("dir") for j in jobs.values()}
    if str(p) not in known or not p.is_dir():
        raise HTTPException(400, "That folder is not available.")
    if hasattr(os, "startfile"):
        os.startfile(p)  # noqa: S606 - opens the folder in Explorer
    return {"path": str(p)}


@app.post("/api/jobs/{jid}/retry")
def retry(jid: str):
    job = get_job(jid)
    with lock:
        if job["status"] in ("queued", "analyzing", "rendering") or job.get("rendering") or job.get("rerender_queued"):
            raise HTTPException(409, "This video is still being processed; try again in a moment.")
        job["child_pid"] = None
        # manual fixes are kept: they are spans and boxes, not tied to the analysis
        job["notice"] = ""
        job["fresh"] = True  # the user asked for a new analysis: do not serve the cached one
        enqueue(job, "analyze")
        save_job(job)
    return public(job)


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    job = get_job(jid)
    with lock:
        cancels[jid].set()
        jobs.pop(jid, None)
        job_faces.pop(jid, None)
        proxy_locks.pop(jid, None)
    # Working files only: the analysis cache, the preview render and, for uploads, our
    # own copy. Source videos added by path and anything already exported are left alone.
    # job.json goes first, so that a folder whose video is still open in the player
    # cannot come back as a job at the next start.
    (JOBDIR / jid / "job.json").unlink(missing_ok=True)
    for _ in range(10):
        shutil.rmtree(JOBDIR / jid, ignore_errors=True)
        if not (JOBDIR / jid).exists():
            break
        time.sleep(0.3)
    if job["origin"] == "upload" and not any(j["src"] == job["src"] for j in jobs.values()):
        src = Path(job["src"])
        if src.parent == UPLOADS:
            for _ in range(20):  # a worker may still hold the file for a moment
                try:
                    src.unlink(missing_ok=True)
                    break
                except PermissionError:
                    time.sleep(0.25)
    return {"ok": True}


@app.get("/api/jobs/{jid}/thumb/{name}")
def thumb(jid: str, name: str):
    get_job(jid)
    f = JOBDIR / jid / Path(name).name
    if not f.exists() or f.suffix != ".jpg":
        raise HTTPException(404)
    return FileResponse(f)


def make_proxy(src: str, dst: Path):
    """The same video and audio streams in a new MP4 with the index (moov) first.
    Zoom writes the index at the end, so a browser has to reach the end of the file
    before it can play or seek, which shows as a black player on long recordings.
    Streams are copied, not re-encoded: a few seconds even for a large file."""
    import av
    tmp = dst.with_name(dst.stem + ".part.mp4")
    with av.open(src) as inp, av.open(str(tmp), "w", options={"movflags": "faststart"}) as out:
        streams = [s for s in inp.streams if s.type in ("video", "audio")]
        omap = {s.index: out.add_stream_from_template(s) for s in streams}
        for pkt in inp.demux(streams):
            if pkt.dts is None:
                continue
            pkt.stream = omap[pkt.stream.index]
            out.mux(pkt)
    os.replace(tmp, dst)


proxy_locks: dict[str, threading.Lock] = {}


@app.get("/api/jobs/{jid}/source")
async def source(jid: str):
    """The recording for the review screen: a browser-friendly copy made on first use
    (and removed again when the video is marked as reviewed), the original if the
    copy cannot be made."""
    job = get_job(jid)
    p = JOBDIR / jid / "source.mp4"
    if not p.exists() and not Path(job["src"]).exists():
        raise HTTPException(404, f"The original video is no longer at {job['src']}")
    if not p.exists():
        lk = proxy_locks.setdefault(jid, threading.Lock())

        def build():
            with lk:
                if not p.exists():
                    try:
                        p.parent.mkdir(parents=True, exist_ok=True)
                        make_proxy(job["src"], p)
                    except Exception:
                        pass
        await asyncio.to_thread(build)
    return FileResponse(p if p.exists() else job["src"])


_tile_finder = None


def tile_around(src: str, t: float, x: float, y: float):
    """The video tile that contains the point (x, y) in the frame at t, found afresh in that
    frame by both finders (the smallest one when nested), or None."""
    import av
    import cv2
    from .detect import find_tiles
    from .tiledet import MODEL_FILE, TileDetector, merge
    global _tile_finder
    if _tile_finder is None and MODEL_FILE.exists():
        _tile_finder = TileDetector()
    with av.open(src) as c:
        v = c.streams.video[0]
        c.seek(max(0, int((t - 0.5) / v.time_base)), stream=v)
        frame = next((f for f in c.decode(v) if f.time is not None and f.time >= t - 1e-3), None)
        if frame is None:
            return None
        bgr = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    tiles = find_tiles(gray)
    if _tile_finder is not None:
        tiles = merge(_tile_finder.detect(bgr, gray), tiles)
    inside = [tl for tl in tiles if tl.x0 <= x <= tl.x1 and tl.y0 <= y <= tl.y1]
    if not inside:
        return None
    tl = min(inside, key=lambda k: (k.x1 - k.x0) * (k.y1 - k.y0))
    return [int(tl.x0), int(tl.y0), int(tl.x1), int(tl.y1)]


@app.get("/api/jobs/{jid}/tile_at")
async def tile_at(jid: str, t: float, x: float, y: float):
    """For the one-click fix: the tile under the clicked point in the frame at t."""
    job = get_job(jid)
    if not job.get("video"):
        raise HTTPException(409, "analysis not finished")
    try:
        box = await asyncio.to_thread(tile_around, job["src"], max(0.0, t), x, y)
    except Exception as e:  # a decode hiccup must not break the page
        raise HTTPException(500, f"Could not read that frame: {e}")
    return {"box": box}


@app.get("/api/jobs/{jid}/tiles")
def tiles(jid: str):
    """Every video tile found at every analysed moment, for the page's snap-to-tile:
    [[t, [[x0, y0, x1, y1, pid], ...]], ...]."""
    get_job(jid)
    f = JOBDIR / jid / "analysis.json"
    if not f.exists():
        raise HTTPException(409, "analysis not finished")
    a = read_json(f)
    return [[s["t"], [[*t["box"], t["pid"]] for t in s["tiles"]]] for s in a["samples"]]


STRIP_N, STRIP_W = 60, 160


def make_strip(src: str, dst: Path):
    """One row of evenly spaced thumbnails of the video (for the range picker)."""
    import av
    import cv2
    cells = []
    with av.open(src) as c:
        v = c.streams.video[0]
        dur = float(v.duration * v.time_base) if v.duration else (c.duration or 0) / 1e6
        h = max(1, round(STRIP_W * v.height / v.width))
        for i in range(STRIP_N):
            t = (i + 0.5) * dur / STRIP_N
            c.seek(int(t / v.time_base), stream=v)
            frame = next(c.decode(v), None)
            if frame is None:
                break
            cells.append(cv2.resize(frame.to_ndarray(format="bgr24"), (STRIP_W, h), interpolation=cv2.INTER_AREA))
    if cells:
        cv2.imwrite(str(dst), cv2.hconcat(cells), [cv2.IMWRITE_JPEG_QUALITY, 70])


@app.get("/api/jobs/{jid}/strip")
async def strip(jid: str):
    job = get_job(jid)
    f = JOBDIR / jid / "strip.jpg"
    if not f.exists() and not Path(job["src"]).exists():
        raise HTTPException(404, f"The original video is no longer at {job['src']}")
    if not f.exists():
        await asyncio.to_thread(make_strip, job["src"], f)
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/jobs/{jid}/output")
def output(jid: str):
    job = get_job(jid)
    if not job["output"] or not Path(job["output"]).exists():
        raise HTTPException(404)
    return FileResponse(job["output"], headers={"Cache-Control": "no-store"})


@app.post("/api/settings")
async def update_settings(request: Request):
    body = await request.json()
    with lock:
        if "experimenters" in body:
            settings["experimenters"] = [str(s).strip() for s in body["experimenters"] if str(s).strip()]
        if "forget_face" in body:
            settings["known_faces"] = [e for e in settings["known_faces"] if e["name"] != body["forget_face"]]
            for job in jobs.values():  # people recognised through that face are unknown again
                for p in job["participants"]:
                    if p.get("experimenter") and p.get("anon") and names_match(p["name"], body["forget_face"]):
                        p["experimenter"] = False
                save_job(job)
        if "out_height" in body:
            settings["out_height"] = max(120, min(1080, int(body["out_height"]))) & ~3  # 4:2:0 planes need a multiple of 4
            for job in jobs.values():
                r = job.get("render") or {}
                if job["status"] == "done" and r.get("height") not in (None, settings["out_height"]) and not job.get("rerender_queued"):
                    job["rerender_queued"] = True
                    tasks.put((job["id"], "rerender"))
        save_settings()
        # a new experimenter list may settle videos that were waiting for a choice
        for job in jobs.values():
            if job["status"] == "choose":
                settle(job)
                save_job(job)
    return public_settings()
