"""Start the local web app:  python run.py  [--port 8756] [--no-browser]

Writes a line to server.log at start, once a minute while alive, and on the way out
(with the reason when the console tells us: window closed, Ctrl+C, logoff). If the log
simply stops without an exit line, the process was killed from outside.
"""
import argparse
import atexit
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

LOG = Path(__file__).resolve().parent / "server.log"


def note(msg: str):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()} {msg}\n")
    except OSError:
        pass


def parent_name() -> str:
    return f"parent pid {os.getppid()}"


def watch_console():
    """Log which console event ends us (Windows): 0 Ctrl+C, 1 Ctrl+Break, 2 window closed,
    5 logoff, 6 shutdown. Returning False lets the default handling proceed."""
    try:
        import ctypes
        HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

        def handler(event):
            note(f"console event {event} ({ {0: 'Ctrl+C', 1: 'Ctrl+Break', 2: 'console window closed', 5: 'logoff', 6: 'shutdown'}.get(event, '?')})")
            return False

        watch_console.keep = HandlerRoutine(handler)  # must stay referenced
        ctypes.windll.kernel32.SetConsoleCtrlHandler(watch_console.keep, True)
    except Exception:
        pass


def heartbeat():
    while True:
        time.sleep(60)
        note("alive")


if __name__ == "__main__":
    # the console may not be UTF-8 (e.g. cp1252 when output is piped); never crash on a message
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8756)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    note(f"start, python {sys.executable}, started by [{parent_name()}]")
    atexit.register(lambda: note("exit (normal shutdown)"))
    watch_console()
    threading.Thread(target=heartbeat, daemon=True).start()
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"\n  Zoom Child Crop is running at {url}\n  Close this window to quit.\n")
    try:
        uvicorn.run("zoomcrop.server:app", host="127.0.0.1", port=args.port, log_level="warning")
    except BaseException as e:
        note(f"exit with {type(e).__name__}: {e}")
        raise
