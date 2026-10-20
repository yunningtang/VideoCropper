# Zoom Child Crop

[中文](README.zh-CN.md)

Cuts one participant's video tile (the child's) out of a Zoom screen recording and writes
it as a fixed-size video. The tile is followed as it is resized, dragged around, or
pushed partly off-screen. Everything runs on your own machine.

## Install

1. Install Python 3.12 from <https://www.python.org/downloads/> (tick "Add python.exe to PATH").
2. Unzip this folder anywhere. The first `start.bat` creates a private Python environment
   in `.venv` and installs the packages from `requirements.txt` (a few minutes, about
   600 MB of downloads). Nothing else is needed: the models are in `zoomcrop/models`.

## Start

Double-click `start.bat`; the browser opens <http://127.0.0.1:8756>. Close the console
window to quit.

## Use

1. Drop videos (or a whole folder) on the left side of the page. You can also paste a
   local path, which processes the files in place without copying them.
2. The rule for telling people apart is "take the experimenters out; whoever is left is
   the child". The first time, it stops and asks you to click the person to crop; the
   other person in that video is remembered as an experimenter, by name when the tile
   has a Zoom name label, by face when it does not (the name can be picked from the
   registered experimenters, which ties the face to the name). From then on, a video
   where only one person remains after the experimenters are taken out runs on its
   own; it asks again only when a third person shows up. When only one person is found
   and nobody was recognised as an experimenter, the video is still cropped but marked
   "Check who was cropped".
   The experimenter list is shown under the picker; click × to remove an entry. To crop
   someone else later, click Change person; this does not change the list.
3. Open a finished video: the original with a red box on the left, the cropped preview
   on the right, playing in sync. On the timeline, tall orange blocks are spans worth
   checking; red hatching means the child's tile was not found.
4. To fix a span: pause where the crop is wrong and click the child's window on the
   original. The window's edges are measured in that very frame and the fix covers the
   span up to the next layout change; it is saved at once and the preview re-exported in
   the background. The red box can also be dragged, or a corner pulled; "No child here"
   and "Keep previous box" cover the other cases, and I / O mark a span of your own.
5. Click Reviewed, next video to move on to the next unreviewed video.
6. Results stay in the tool's own workspace until you click Export… and choose which
   videos, which folder (Browse… opens the system folder dialog) and which files. If a
   file with the same name exists, you are asked before it is replaced.

Keyboard shortcuts in the review screen: Space play/pause, ← → 1 s (Shift: 5 s),
`,` `.` one frame, P / N previous / next span to check.

## Batches of videos

- Add them by path (paste the folder), not by dropping: dropped files are copied into the
  workspace, pasted paths are read in place. Either way the folder's name (B01, B02, ...)
  is kept as the videos' batch: the browser can filter by it, the summary CSV has a column
  for it, and Export… can write one subfolder per batch.
- Manual fixes are saved at once and their preview files re-rendered in the background,
  ahead of waiting analyses. Closing the console window and starting again later loses
  nothing that was saved: queued videos resume, interrupted analyses start over, and
  pending re-renders are picked up.
- Register the experimenters first (name or face), or every video stops to ask who the
  child is.
- Four videos run at a time, each in its own process. A video whose analysis fails is
  retried once, then marked Failed with the reason in its `error.log`; the others carry
  on. Analysis results are cached by video content (`workspace/cache`), so a video that
  is removed and added again, or re-added after a restart, is not analysed twice.
- With a graphics card, run `tools\gpu.bat` once: the tile model then runs on the GPU
  and analysis is roughly twice as fast. Without it, everything runs on the CPU. Run it
  again after any `pip install -r requirements.txt` (that puts the CPU build back).
- `server.log` records when the server started, a heartbeat every minute, and why it
  stopped (window closed, Ctrl+C, logoff); a log that just stops means the process was
  killed from outside.

## Exported files

| File | Contents |
| --- | --- |
| `name_child.mp4` | The crop. Frame timestamps match the source and the audio is copied as-is, so it lines up with the original. |
| `name_segments.csv` | Start, end, crop box, status and flags of every segment. |
| `name_report.json` | The same, plus the participant list, statistics and manual fixes (off by default). |
| `crop_summary.csv` | One row per exported video: visible share, spans to check, manual fixes, reviewed or not. |

Spans without the child are black in the output; so is the part of a tile that was
dragged off-screen (the picture keeps its position).

## How it works

A fuller description, step by step with the models and thresholds, is in
[docs/pipeline.md](docs/pipeline.md).


- **Finding tiles**: two finders combined. The geometric one knows Zoom tiles are 16:9
  rectangles: it looks for long straight edges, assembles rectangles and requires each
  side to be backed by enough edge pixels; when the border is visible its boxes are
  pixel-exact. The learned one (`zoomcrop/models/tiles.onnx`, a small YOLO model)
  recognises "this region is a camera picture" without relying on the border, so a tile
  half-covered by a context menu, with the mouse over it, or with white clothing against
  a white slide is still found; its boxes are then snapped to the nearest real edge.
  Where both agree the geometric box is used; a tile only the model saw uses the
  model's box. Without the model file, only the geometric finder runs.
- **Following tiles**: every 0.5 s; a tile that stays in place (or glides while being
  dragged) continues the same track. The track ends when the picture inside changes all
  at once (Zoom reseated people). When the layout changes between two samples, the
  frames in between are examined to find the exact moment. Borders a single frame shows
  badly (a thin separator between two dark pictures) are recovered from an edge map
  averaged over the last few seconds.
- **People**: tracks that are on screen at the same time are never the same person;
  tracks with the same face, the same slot, or the same look are chained. A chain is a
  person when its picture keeps changing the way live video does, or when a face was
  seen in it; boxes on slides are filtered out this way.
- **Names**: the Zoom name label is read (OCR) when there is one; otherwise the face is
  compared with the remembered experimenters; otherwise Person A, Person B.
- **Elimination**: when the child's face is covered, turned away or too close, that tile
  cannot be recognised on its own. The other tiles in the frame decide: if the
  experimenter's tile is recognised, the remaining live tile is the child's (flagged
  `inferred`, not for review). If nobody is recognised but a tile sits where the child's
  was a moment ago, it is taken as the child's and flagged `held` for review. A single
  remaining tile with nobody recognised is left out and flagged `unsure`.

## Known limits

- Recordings without name labels (1440×900 Mac recordings) tell people apart by look and
  face; the first one asks who the child is. With "remember by face" ticked, later
  videos with the same experimenter do not ask. Only the face description (a vector of
  numbers) is kept, never a picture. Between analysis and the choice, every
  participant's vectors sit in the job's workspace folder (`faces.json`); the child's are
  deleted the moment the child is chosen. Old records are discarded when the face model
  version changes.
- Two children in one video, or a parent and a child in separate tiles: the tool cannot
  tell which is the child and asks.
- Two display names that differ by one letter may be taken for the same person.
- When the child turns the camera off, the tile stays and is cropped as usual (flagged
  "camera may be off" in the report).
- Validated on five sample recordings (1920×1080 and 1440×900, shared slides plus video
  tiles); `python tests/regress.py` re-runs them against hand-checked boxes. Other
  layouts (gallery view, non-16:9 tiles) are untested: try one or two videos before a
  batch.

## Retraining the tile detector

No boxes have to be drawn by hand. Labels come from the tool's own checked timelines
(including the frames it bridged), the hand-checked boxes in `tests/truth.json` (the
validation set), and synthetic frames with tiles pasted under menus, cursors and title
bars.

```
python -m venv .venv-train
.venv-train\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv-train\Scripts\pip install ultralytics onnx av opencv-python
.venv\Scripts\python tools\dataset.py          # exports data/tiles (analyses unseen videos first)
.venv-train\Scripts\python tools\train.py      # trains and exports zoomcrop/models/tiles.onnx
python tests\regress.py                         # checks against the hand-checked boxes
```

The more videos are processed and reviewed, the more labels the next export yields;
every Manual fix becomes training data too. The app itself needs only onnxruntime,
not torch.

## Files

```
run.py / start.bat      start the app
tools/dataset.py        export detector training data from analysed videos
tools/train.py          train the detector and export onnx
zoomcrop/tiledet.py     learned tile finder (onnxruntime), merged with the geometric one
zoomcrop/detect.py      find 16:9 video tiles in one frame
zoomcrop/analyze.py     whole-video analysis: sampling, tracking, names, faces, people
tests/regress.py        regression test (tests/truth.json holds the hand-checked answers)
zoomcrop/timeline.py    crop segments, flags, manual fixes
zoomcrop/render.py      crop and encode
zoomcrop/work.py        runs analysis and rendering in child processes
zoomcrop/server.py      local web server
zoomcrop/static/        the page
workspace/              analysis cache, uploaded copies, settings (safe to delete)
```
