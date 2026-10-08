"""Start the local web app:  python run.py  [--port 8756] [--no-browser]"""
import argparse
import sys
import threading
import webbrowser

import uvicorn

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
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"\n  Zoom Child Crop is running at {url}\n  Close this window to quit.\n")
    uvicorn.run("zoomcrop.server:app", host="127.0.0.1", port=args.port, log_level="warning")
