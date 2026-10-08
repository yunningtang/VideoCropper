"""Pass 2: crop every frame to its segment's box and encode at a fixed size.

Frames keep their original timestamps and the audio stream is copied as-is, so
the output stays frame-for-frame aligned with the source recording.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import av
import cv2
import numpy as np

from .analyze import Cancelled

AUDIO_COPY_OK = {"aac", "mp3", "ac3", "alac", "opus"}


def _plane(frame: av.VideoFrame, i: int) -> np.ndarray:
    p = frame.planes[i]
    return np.frombuffer(p, np.uint8).reshape(p.height, p.line_size)[:, :p.width]


def _plan(box, W, H, out_w, out_h, inset, cut=None):
    """Source and destination rectangles (even-aligned, for 4:2:0) for one crop box.
    cut: the tile is hidden below this y (the floating panel ends there)."""
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = x0 + inset, y0 + inset, x1 - inset, y1 - inset
    w, h = x1 - x0, y1 - y0
    vx0, vy0, vx1, vy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cut is not None:
        vy1 = min(vy1, cut - inset)
    if vx1 - vx0 < 4 or vy1 - vy0 < 4:
        return None
    # the part of the box that is off-screen stays black in the output
    dx0 = int(round((vx0 - x0) / w * out_w)) & ~1
    dy0 = int(round((vy0 - y0) / h * out_h)) & ~1
    dx1 = (int(round((vx1 - x0) / w * out_w)) + 1) & ~1
    dy1 = (int(round((vy1 - y0) / h * out_h)) + 1) & ~1
    dx1, dy1 = min(dx1, out_w), min(dy1, out_h)
    sx0, sy0 = vx0 & ~1, vy0 & ~1
    sx1, sy1 = min(W & ~1, (vx1 + 1) & ~1), min(H & ~1, (vy1 + 1) & ~1)
    if dx1 - dx0 < 2 or dy1 - dy0 < 2:
        return None
    return (sx0, sy0, sx1, sy1), (dx0, dy0, dx1, dy1)


def render(src: str | Path, dst: str | Path, segments: list[dict], out_h: int = 360,
           crf: int = 20, inset: int = 2,
           progress: Callable[[float], None] | None = None,
           cancel: Callable[[], bool] | None = None) -> dict:
    out_h = int(out_h) & ~1
    out_w = int(round(out_h * 16 / 9)) & ~1
    dst = Path(dst)
    tmp = dst.with_name(dst.stem + ".part" + dst.suffix)
    inp = av.open(str(src))
    out = av.open(str(tmp), "w", options={"movflags": "faststart"})
    info = {"audio": False, "frames": 0, "width": out_w, "height": out_h}
    try:
        iv = inp.streams.video[0]
        iv.thread_type = "AUTO"
        W, H = iv.width, iv.height
        duration = float(iv.duration * iv.time_base) if iv.duration else (inp.duration or 0) / 1e6
        ov = out.add_stream("libx264", rate=iv.average_rate or 25)
        ov.width, ov.height, ov.pix_fmt = out_w, out_h, "yuv420p"
        ov.codec_context.time_base = iv.time_base
        ov.options = {"crf": str(crf), "preset": "veryfast"}
        for attr in ("colorspace", "color_range", "color_primaries", "color_trc"):
            try:
                setattr(ov.codec_context, attr, getattr(iv.codec_context, attr))
            except Exception:
                pass

        ia = next((a for a in inp.streams.audio), None)
        oa = None
        if ia is not None and ia.codec_context.name in AUDIO_COPY_OK:
            oa = out.add_stream_from_template(ia)
            info["audio"] = True
        streams = [iv] + ([ia] if oa is not None else [])

        black = np.empty((out_h * 3 // 2, out_w), np.uint8)
        black[:out_h] = 16
        black[out_h:] = 128
        plans = [(_plan(s["box"], W, H, out_w, out_h, inset, s.get("cut")) if s["box"] else None) for s in segments]
        si = 0
        hw, hh = out_w // 2, out_h // 2

        for packet in inp.demux(streams):
            if packet.stream.type == "audio":
                if packet.dts is None:
                    continue
                packet.stream = oa
                out.mux(packet)
                continue
            for frame in packet.decode():
                t = frame.time or 0.0
                while si + 1 < len(segments) and t >= segments[si + 1]["start"] - 1e-6:
                    si += 1
                plan = plans[si] if segments else None
                buf = black.copy()
                if plan is not None:
                    if frame.format.name != "yuv420p" or frame.width % 2 or frame.height % 2:
                        frame = frame.reformat(width=W & ~1, height=H & ~1, format="yuv420p")
                    (sx0, sy0, sx1, sy1), (dx0, dy0, dx1, dy1) = plan
                    dw, dh = dx1 - dx0, dy1 - dy0
                    shrink = (sx1 - sx0) > dw
                    # read the crop straight out of the decoder's planes (no full-frame copy)
                    y = _plane(frame, 0)[sy0:sy1, sx0:sx1]
                    buf[dy0:dy1, dx0:dx1] = cv2.resize(
                        y, (dw, dh), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)
                    for k in range(2):
                        c = _plane(frame, k + 1)[sy0 // 2:sy1 // 2, sx0 // 2:sx1 // 2]
                        plane = buf[out_h + k * out_h // 4: out_h + (k + 1) * out_h // 4].reshape(hh, hw)
                        plane[dy0 // 2:dy1 // 2, dx0 // 2:dx1 // 2] = cv2.resize(
                            c, (dw // 2, dh // 2), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
                of = av.VideoFrame.from_ndarray(buf, format="yuv420p")
                of.pts = frame.pts
                of.time_base = iv.time_base
                for p in ov.encode(of):
                    out.mux(p)
                info["frames"] += 1
                if info["frames"] % 50 == 0:
                    if cancel and cancel():
                        raise Cancelled()
                    if progress and duration > 0:
                        progress(min(1.0, t / duration))
        for p in ov.encode():
            out.mux(p)
        out.close()
        inp.close()
        os.replace(tmp, dst)
    except BaseException:
        out.close()
        inp.close()
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return info
