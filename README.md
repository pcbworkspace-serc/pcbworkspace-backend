# PCBWorkspace SERC backend

Flask backend for PCBWorkspace (`flask_server.py`). Run the tests with
`python -m pytest`.

## Vision held-out evaluation (`vision_eval.py`)

**Where the numbers stand.** No detection or alignment accuracy has been
measured on SERC boards yet. `/nn/status` says so in
`detection_metrics.serc_boards`. The only detection figures it carries are
in `detection_metrics.fpic_paper_reference`: they are published numbers from
the FPIC paper, measured on the public FPIC dataset. The align/validate
heads' `synthetic_val_metrics` come from synthetic renders. Don't quote
either one as SERC performance. `vision_eval.py` produces the first SERC
number once people have labeled real frames.

**1. Collect frames.** Start the server with `CAPTURE_TRAINING_DATA=1`. Send
`board_id` and `session_id` as form fields with every `/nn/align`,
`/nn/detect_boxes` or `/nn/detect_boxes_yolo` request. Each frame is saved
under `CAPTURE_DIR/<kind>/` together with its `capture_id`, and with the live
`px_per_mm` if a calibration is saved.

**2. Label them yourself.** Send `POST /nn/feedback` with
`{"kind", "capture_id", "label_source": "human", "labeler": "<your id>", "ground_truth": ...}`.
Draw or measure the label independently; don't just accept the prediction
shown to you, because the harness refuses labels identical to the model's
output.
- `detect`: `{"boxes": [{"class": "R", "box": [x1, y1, x2, y2]}]}` in image
  pixels, using the class codes in `pcb_jepa_nn.CLASS_NAMES`. Send `[]` for
  a frame with no parts.
- `align`: `{"delta_theta_deg", "delta_x_mm", "delta_y_mm"}` (or `_px` with
  an optional `px_per_mm`). Offsets are the part center minus the image
  center, with x to the right and y down. The angle is positive clockwise
  as seen on screen. This is the same convention `/nn/align` returns.

**3. Run it.**

```bash
python vision_eval.py --capture-dir data_capture --out vision_eval_report.json --write-split vision_eval_split.json
```

The report gives:
- per-class and pooled precision/recall for the YOLOv8n + MobileNetV3
  hybrid, with `--detectors yolo_hybrid,sliding_window` to add the sliding
  window. It needs `ultralytics`; without it the detector is reported
  `not_run`.
- angle error (deg) and center-offset error (mm) for the classical-CV and
  JEPA alignment paths on the same frames, plus each path's coverage and
  the paired JEPA − classical difference.

Every proportion gets a 95% Wilson interval, and every statistic gets a 95%
bootstrap interval that resamples whole boards/sessions. The Wilson interval
treats boxes as independent, so it runs narrow when boxes share a board. With
fewer than 5 held-out groups, the bootstrap resamples frames instead and the
report warns that the interval is too narrow.

**What it refuses to report** (exit code 2, no file written):
- held-out labels whose `label_source` isn't `"human"`, that have no
  labeler, or that copy the model's own prediction
- a held-out board/session or capture listed in `--trained-on`
- one image filed under two boards/sessions
- malformed labels or missing images
- synthetic and real captures mixed in one dir

A dir of synthetic fixtures (`capture_context.synthetic`, as the tests
write) is evaluated, but the report is stamped `SYNTHETIC_TEST_FIXTURE`.

**Split.** Frames that share a `board_id` or `session_id` form one group,
and whole groups are held out (default 30%, set with `--test-fraction`).
Frames are never split individually, because near-identical frames of one
board would leak across. Frames with neither id are left out and counted.
A group's side depends only on its ids and `--split-seed`, so adding data
never moves a held-out board into the reserved set. Fine-tuning should use
only `train_reserved` from `--write-split` and record the ids it trained on.
Pass that record as `--trained-on` on the next evaluation.

**When quoting a number,** state the conditions from the report's
`conditions` block:
- `--iou` (default 0.5) and `--score-threshold` (default 0, i.e. the
  endpoint's own YOLO confidence cutoff)
- `--angle-period` (default 360). Use 180 for parts without a visible
  polarity mark: on a plain rectangle the JEPA head can be 180° off while
  being right up to symmetry.
- where `px_per_mm` came from (`px_per_mm_sources`). `--px-per-mm` is only
  a fallback and must be given together with `--px-per-mm-source`.
- the model checksums in `models`
- how many frames and boards were held out
