"""Analysis and rendering run in child processes.

The web server only schedules and shows progress. Each task gets a fresh process: a
crash inside a native library (decoder, ONNX runtime) takes down that task alone, the
server's event loop is never starved by CPU-bound work, and the thread count per process
stays what the libraries were configured for. Progress, the result and any error come
back through a queue; cancelling kills the process.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import queue
import traceback
from pathlib import Path
from typing import Callable

from .analyze import Cancelled


class WorkerFailed(Exception):
    pass


# -- child-side targets (module level, so that "spawn" can import them) ----------------
def analyze_task(src: str, jdir: str, known_faces: list, q):
    try:
        from .analyze import Analyzer
        an = Analyzer(known_faces=known_faces)
        res = an.run(src, thumb_dir=jdir, progress=lambda p: q.put(("progress", float(p))))
        Path(jdir, "analysis.json").write_text(json.dumps(res), encoding="utf-8")
        q.put(("done", {"participants": res["participants"], "video": res["video"], "faces": an.face_vectors}))
    except BaseException:
        q.put(("error", traceback.format_exc()))


def render_task(src: str, dst: str, segments: list, out_h: int, q):
    try:
        from .render import render
        info = render(src, dst, segments, out_h=out_h, progress=lambda p: q.put(("progress", float(p))))
        q.put(("done", info))
    except BaseException:
        q.put(("error", traceback.format_exc()))


# -- server side -----------------------------------------------------------------------
def run(target: Callable, args: tuple, progress: Callable[[float], None], cancel: Callable[[], bool]):
    """Run target(*args, queue) in a child process. Relays progress, returns the payload
    of its "done" message, raises Cancelled when cancel() turns true (the process is
    killed) and WorkerFailed when the task reports an error or the process dies."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(*args, q), daemon=True)
    p.start()
    try:
        while True:
            try:
                kind, val = q.get(timeout=0.5)
            except queue.Empty:
                if cancel():
                    raise Cancelled()
                if p.is_alive():
                    continue
                try:  # the last message may still be in flight
                    kind, val = q.get(timeout=2.0)
                except queue.Empty:
                    raise WorkerFailed(f"The worker process stopped unexpectedly (exit code {p.exitcode}). "
                                       "See server.err.log for what the libraries printed.")
            if kind == "progress":
                progress(val)
            elif kind == "done":
                return val
            else:
                raise WorkerFailed(val)
    finally:
        if p.is_alive():
            p.terminate()
        p.join(5)
        q.close()
