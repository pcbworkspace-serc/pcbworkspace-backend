"""
data_capture.py — turns real /nn/align and /nn/validate traffic into a
labeled dataset over time, so the sim-to-real gap flagged in jepa_heads.py
can actually get closed with real data instead of staying a permanent
caveat.

Important: logging the image + the model's own prediction is NOT useful
training data by itself — you can't train a model on its own guesses, that
teaches it nothing. The prediction is captured for context (what did the
model think at the time), but the thing that makes a capture usable for
retraining is a *ground truth* label attached later via record_feedback()
— from a human reviewing the placement, or from whatever downstream signal
your pick-place loop already has for "did this actually work."

Two-part flow:
  save_capture()      — called from flask_server.py on every /nn/align,
                         /nn/validate or /nn/detect_boxes[_yolo] request when
                         CAPTURE_TRAINING_DATA is set. Writes the image + a
                         JSON sidecar, returns a capture_id.
  record_feedback()   — called from the new /nn/feedback endpoint once the
                         real outcome is known. Appends ground_truth to the
                         matching sidecar.

Two things a held-out evaluation needs that the image and label alone
don't say, so the sidecar records them too:
  board_id / session_id  — which physical board and which capture session a
                         frame came from. Frames of one board are near
                         duplicates of each other, so vision_eval.py splits
                         by these, never by frame. Either can be sent with
                         the capture request or filled in later by feedback.
  label_source / labeler — who produced ground_truth: "human" (with the
                         labeler's id), "model", or "automated" (a downstream
                         pick-place signal). vision_eval.py reports only on
                         labels a person made; a label copied from the model
                         would just measure the model against itself.

load_labeled_dataset() reads back every capture that has a ground_truth
attached — that's the set usable for a future fine-tuning pass, and what
vision_eval.py evaluates on. This is the collection half of the loop, not a
retraining pipeline.
"""

import json
import time
import uuid
from pathlib import Path

DEFAULT_CAPTURE_DIR = "data_capture"
KINDS = ("align", "validate", "detect")
LABEL_SOURCES = ("human", "model", "automated")


def _capture_dir(kind, base_dir=DEFAULT_CAPTURE_DIR):
    d = Path(base_dir) / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_kind(kind):
    if kind not in KINDS:
        raise ValueError("kind must be one of " + ", ".join(repr(k) for k in KINDS))


def _clean_id(value):
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def save_capture(kind, pil_image, prediction, base_dir=DEFAULT_CAPTURE_DIR,
                 board_id=None, session_id=None, context=None):
    """kind: 'align', 'validate' or 'detect'. context holds conditions of
    the capture an evaluation needs later (e.g. px_per_mm from the live
    calibration). Returns the capture_id."""
    _check_kind(kind)

    capture_id = uuid.uuid4().hex[:12]
    d = _capture_dir(kind, base_dir)
    ts = time.time()

    img_path = d / f"{capture_id}.png"
    pil_image.save(img_path, format="PNG")

    sidecar = {
        "capture_id": capture_id,
        "kind": kind,
        "captured_at": ts,
        "image_file": img_path.name,
        "board_id": _clean_id(board_id),
        "session_id": _clean_id(session_id),
        "capture_context": context or {},
        "model_prediction": prediction,
        "ground_truth": None,
        "label_source": None,
        "labeler": None,
        "labeled_at": None,
    }
    with open(d / f"{capture_id}.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    return capture_id


def record_feedback(kind, capture_id, ground_truth, base_dir=DEFAULT_CAPTURE_DIR,
                    label_source=None, labeler=None, board_id=None, session_id=None):
    """Attaches a real-world outcome to a previously saved capture.
    Returns the updated sidecar dict, or None if the capture_id doesn't exist.

    label_source: 'human' | 'model' | 'automated', or None when the caller
    didn't say (stored as None; vision_eval.py treats that as not human).
    A 'human' label must name its labeler. board_id / session_id fill in a
    capture that was saved without them; they may not change one already set.
    Raises ValueError on any of those violations."""
    _check_kind(kind)
    if label_source is not None and label_source not in LABEL_SOURCES:
        raise ValueError("label_source must be one of " + ", ".join(repr(s) for s in LABEL_SOURCES))
    labeler = _clean_id(labeler)
    if label_source == "human" and not labeler:
        raise ValueError("a human label must name its labeler")

    sidecar_path = _capture_dir(kind, base_dir) / f"{capture_id}.json"
    if not sidecar_path.exists():
        return None

    with open(sidecar_path) as f:
        sidecar = json.load(f)
    for key, value in (("board_id", _clean_id(board_id)), ("session_id", _clean_id(session_id))):
        if value is None:
            continue
        if sidecar.get(key) not in (None, value):
            raise ValueError(f"{key} {value!r} conflicts with {sidecar[key]!r} recorded at capture time")
        sidecar[key] = value
    sidecar["ground_truth"] = ground_truth
    sidecar["label_source"] = label_source
    sidecar["labeler"] = labeler
    sidecar["labeled_at"] = time.time()
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    return sidecar


def load_labeled_dataset(kind, base_dir=DEFAULT_CAPTURE_DIR):
    """Every capture that has a ground_truth attached - the subset that's
    actually usable for a future retraining pass. Everything else is a
    capture still waiting to be labeled. Read-only: doesn't create the
    kind's folder, so evaluating someone's capture dir leaves it as found."""
    _check_kind(kind)
    d = Path(base_dir) / kind
    labeled = []
    for sidecar_path in sorted(d.glob("*.json")):
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        if sidecar.get("ground_truth") is not None:
            sidecar["image_path"] = str(d / sidecar["image_file"])
            labeled.append(sidecar)
    return labeled


def capture_stats(base_dir=DEFAULT_CAPTURE_DIR):
    stats = {}
    for kind in KINDS:
        d = _capture_dir(kind, base_dir)
        total = len(list(d.glob("*.json")))
        labeled = len(load_labeled_dataset(kind, base_dir))
        stats[kind] = {"total_captured": total, "labeled": labeled, "unlabeled": total - labeled}
    return stats
