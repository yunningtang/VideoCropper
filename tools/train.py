"""Train the tile detector and export it for the app.

    .venv-train/Scripts/python tools/train.py [--data data/tiles/data.yaml] [--epochs 40] [--imgsz 960]

Needs the training environment (.venv-train: torch + ultralytics). The result goes to
zoomcrop/models/tiles.onnx, which zoomcrop/tiledet.py loads with onnxruntime; the app
itself never needs torch.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "tiles" / "data.yaml"))
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="tiles")
    args = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.model)
    # Tiles are axis-aligned screen regions: no flips or hue tricks that make no sense
    # for screenshots; scale and translate jitter keep sizes and positions varied.
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name,
                project=str(ROOT / "runs"), exist_ok=True, fliplr=0.0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,
                mosaic=0.5, scale=0.4, translate=0.1, degrees=0.0, shear=0.0, perspective=0.0,
                close_mosaic=5, patience=15, workers=4, seed=0, verbose=False)
    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, plots=False, verbose=False)
    print(f"val: mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}  precision {metrics.box.mp:.3f}  recall {metrics.box.mr:.3f}")
    onnx = Path(YOLO(str(best)).export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True, opset=17))
    dst = ROOT / "zoomcrop" / "models" / "tiles.onnx"
    shutil.copyfile(onnx, dst)
    print(f"exported {dst} ({dst.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
