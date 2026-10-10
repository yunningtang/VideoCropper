# How the crop is made

Zoom Child Crop turns a screen recording of a Zoom session into a video that contains
only the child's camera tile. It works in two passes joined by a timeline: the first pass
reads the recording and decides, for every moment, where the child's tile is; the second
pass cuts every frame according to that timeline. People can correct the timeline between
the two passes.

## In short

```
Every 0.5 s take one frame:
  1. Find tiles: geometric finder (16:9 borders) + small model, merged and de-duplicated.
  2. Link tracks: a box that continues the previous frame's position (dragging included)
     joins the same track; a tile whose picture changes all at once ends it.
     When the layout changes between two samples, the frames in between are examined
     to find the exact frame of the change.
  3. Identify: along each track, read the Zoom name label, else compare faces; the
     track's identity is decided by vote over the whole track.
     Take the experimenters out; the remaining track is the child. In a single frame
     where the child was not recognised but the experimenter is on screen, the
     remaining live tile counts as the child's.

First run: the user clicks the child; the other person is stored as an experimenter
(name, or face vector) and reused afterwards; a third person triggers the question again.

Output: a timeline of segments "from s to s, use this box", each with a mark
(green / orange = check / red = not on screen).

Crop: frame by frame, take the current segment's box (fixed within a segment, no
interpolation), cut it out, scale to a fixed size, keep the original timestamps and
copy the audio unchanged.
```

## Pass 1: analysis (`zoomcrop/analyze.py`)

The whole recording is decoded once. Every frame contributes to a running average of
edge maps (cheap), and every 0.5 s one frame is examined in full.

### Finding tiles

Two finders run on each examined frame and their results are merged
(`zoomcrop/detect.py`, `zoomcrop/tiledet.py`).

- **Geometric.** Zoom draws video tiles as 16:9 rectangles. Pixel differences give
  horizontal and vertical edge maps; long straight runs become candidate sides; two
  vertical sides and one horizontal side propose a rectangle, and the proposal is kept
  when each side is backed by edge pixels along at least 60% of its length. A tile
  dragged partly off-screen, or one under the floating window's title bar, is accepted
  on three sides. Boxes are pixel-exact when the border is visible.
- **Learned.** A YOLO11n model trained on this tool's own recordings recognises "a camera
  picture" regardless of its border, so a tile half-covered by a context menu, with the
  mouse over it, or with white clothing against a white slide is still found. Its boxes
  are a few pixels loose and are snapped to the nearest real edge.
- **Merge.** Where a geometric box overlaps a learned box by at least half, the geometric
  box is used (measured edges beat regressed ones). A tile only the model saw keeps the
  model's box. A geometric box that overlaps a learned box by a quarter or more without
  being it is a phantom (a rectangle straddling two neighbouring tiles, or a piece of one)
  and is dropped.

### Linking tracks

A tile that keeps its place from one examined frame to the next (overlap of at least 85%,
or a shift of a few pixels) continues the same track; while a window is dragged, an
overlap of 60% with a matching thumbnail is enough. The track ends when the picture inside
changes all at once and stays changed: Zoom put someone else into that slot. A hand waved
in front of the camera also changes everything for a moment, but the picture before and
after still match, so the track survives.

When the layout differs between two examined frames (a tile appeared, vanished or moved),
the frames in between are searched to find the exact frame of the change: a binary search
for an instant switch, a few evenly spaced frames while a tile glides.

Borders that a single frame shows badly (a thin separator between two dark pictures) are
recovered from the edge map averaged over the last few seconds: the border stays put while
the people move.

### Who is who

- **Chains.** Tracks that are on screen at the same time can never be the same person and
  are never joined. The rest are joined by evidence in order of strength: the same name
  label, the same face, the same slot with a similar look, a similar look. A chain counts
  as a person when its picture keeps changing the way live video does (sensor noise, small
  movements) or when a face was seen in it; boxes on slides fail both tests. A chain seen
  for less than 15% of the video with no face is a scrap, not a person.
- **Names.** The Zoom name label in a tile's bottom-left corner is read by OCR; the text
  is not re-read while its pixels are unchanged. Without a label, the faces along the
  chain are compared with the remembered experimenters. Otherwise the person is
  Person A, Person B, ...
- **The child.** Experimenters are known by name (a list) or by face (a library of face
  vectors). They are taken out; if exactly one substantial person remains, that is the
  child and the video runs on its own. If nobody was recognised as an experimenter, the
  single remaining person is still cropped but the video is marked "check who was
  cropped". Two or more remaining people (a third window) make the tool ask. The first
  time, the user clicks the child, and the other person is remembered as an experimenter,
  by name when the tile has a label, by face when it does not.

Only face vectors of experimenters are ever stored. A child's face description is deleted
the moment the child is chosen.

## The timeline (`zoomcrop/timeline.py`)

For every examined moment the child's box is decided:

- the child's own tile when it was recognised;
- otherwise, by elimination: with the experimenter's tile recognised in the same frame,
  the remaining live tile is the child's (`inferred`, not for review);
- otherwise, by continuity: a live tile sitting where the child's was a moment ago is
  taken as the child's and marked `held` for review;
- a lone unrecognised tile with nobody else recognised is left out and marked `unsure`;
- no tile at all: the child is off screen.

Consecutive moments with the same box (within a few pixels) form one segment; a segment
holds one review state. A gap of up to 1.5 s with the same box on both sides is bridged
silently (the detector blinked); a gap of up to 0.6 s with a different box after it is
bridged and marked for review. Manual fixes (a box, "no child", or "keep the previous box"
over a span) are applied last and override everything.

The result is a list of segments: start, end, box, status and flags. It is what the
review screen shows and what the renderer follows.

## Pass 2: rendering (`zoomcrop/render.py`)

The recording is decoded again, this time every frame. Each decoded frame looks up the
segment covering its timestamp, the box is cut straight out of the decoder's YUV planes,
scaled to the output size (640×360 by default), and encoded with x264. The output frame
carries the source frame's timestamp and time base unchanged; the audio stream is copied
packet for packet without decoding. Spans without a box produce black frames.

Because one output frame is written per decoded source frame with the source's own
timestamp, the cropped file has the same number of frames and the same timestamp on every
frame as the original, including recordings with an unstable frame rate (dropped frames
stay dropped at the same positions). Frame N of the original is frame N of the crop, and
labels made in seconds on the original apply to the crop unchanged. This was verified
frame by frame on recordings at 20, 25 and 30 fps, one of them averaging 29.91 fps with
11 dropped-frame spots.

## Models and methods, step by step

| Step | What | Model / method | Details |
|---|---|---|---|
| Decode | one frame every 0.5 s | PyAV (FFmpeg) | sequential decode, per-frame timestamps |
| Find tiles (a) | 16:9 rectangles by border | geometric, no model | pixel-difference edges → straight segments → two vertical sides plus one horizontal → each side backed by ≥ 60% edge pixels |
| Find tiles (b) | "this is a camera picture" | **YOLO11n** (self-trained, 10 MB) | 960 px input, confidence ≥ 0.35, boxes snapped to the nearest real edge; DirectML on the GPU when available |
| Merge | one box per tile | rule | overlap ≥ 50% → geometric box; model-only → model box; a geometric box overlapping a model box by ≥ 25% without being it is dropped |
| Link tracks | same tile over time | rule | box overlap ≥ 85% or shift ≤ 6 px continues; while dragged, overlap ≥ 60% with thumbnail correlation ≥ 0.5 continues; correlation < 0.3 that persists ends the track |
| Refine | exact frame of a layout change | rule | binary search; a few evenly spaced frames while a tile is dragged |
| Read names | Zoom label text | **RapidOCR** (detector + recogniser) | only the strip in the tile's bottom-left corner; not re-read while the pixels are unchanged |
| Find faces | largest face in the tile | **YuNet** (OpenCV, 0.2 MB) | tile upscaled to 480 px wide before detection |
| Match faces | same person or not | **SFace** (OpenCV, 38 MB) | 128-d vector, cosine ≥ 0.45 same, < 0.25 different |
| Chain | tracks at different times that are one person | rule (union-find) | tracks on screen together are never joined; join by same name → same face → same slot and similar look → similar look |
| Drop non-people | slide boxes are not people | rule | a chain must keep moving like live video (thumbnail change ≥ 0.35 in at least half of the pairs) or have shown a face; share < 15% with no face is dropped |
| Pick the child | whom to crop | rule | remove experimenters (name list or face library); one left → crop; two or more → ask |
| Box per frame | child's box + mark | rule | recognised → its box; else by elimination (inferred) or by position continuity (held); neither → unsure |
| Tidy timeline | merge, bridge gaps | rule | boxes within 5 px merge; a gap ≤ 1.5 s with the same box on both sides is bridged silently; a gap ≤ 0.6 s with a different box after it is bridged and marked |
| Crop and encode | output video | PyAV + **x264** | crop the YUV planes, scale to 640×360, crf 20, timestamps copied, audio stream copied |

Four models: YOLO11n, RapidOCR, YuNet, SFace. Everything else is rules; the thresholds
are constants at the top of each module.

## How the tile model is trained

No boxes are drawn by hand. `tools/dataset.py` exports frames and labels from the tool's
own timelines (moments it was sure about, and moments it bridged, which are exactly what
the geometric finder misses), uses the hand-checked boxes in `tests/truth.json` as the
validation set, and synthesises extra frames by pasting tiles under context menus,
cursors, title bars and into strips. `tools/train.py` trains YOLO11n and exports the
ONNX model the app loads. The more recordings are processed and reviewed, the more labels
the next export yields.

## What has been checked

Five recordings (1920×1080 and 1440×900; shared slides with video tiles), 107 hand-marked
boxes at 61 moments: the child's tile was right at 60 of 61 moments, the experimenter was
never cropped, three of the five videos needed no manual checking, the other two a few
seconds. Single-frame tile finding: 106 of 107 boxes (geometric finder alone: 103).
Other layouts (gallery view, non-16:9 tiles) and other experimenters are untested; a
batch should start with a mixed sample of 10 to 20 recordings.
