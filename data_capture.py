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
  save_capture()      — called from flask_server.py on every /nn/align or
                         /nn/validate request when CAPTURE_TRAINING_DATA is
                         set. Writes the image + a JSON sidecar, returns a
                         capture_id.
  record_feedback()   — called from the new /nn/feedback endpoint once the
                         real outcome is known. Appends ground_truth to the
                         matching sidecar.

load_labeled_dataset() reads back every capture that has a ground_truth
attached — that's the set actually usable for a future fine-tuning pass.
Nothing in this repo consumes it yet (there's no real labeled data to
consume on day one) — this is the collection half of the loop, not a
retraining pipeline.
"""

import json
import time
import uuid
from pathlib import Path

DEFAULT_CAPTURE_DIR = "data_capture"


def _capture_dir(kind, base_dir=DEFAULT_CAPTURE_DIR):
    d = Path(base_dir) / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_capture(kind, pil_image, prediction, base_dir=DEFAULT_CAPTURE_DIR):
    """kind: 'align' or 'validate'. Returns the capture_id."""
    if kind not in ("align", "validate"):
        raise ValueError("kind must be 'align' or 'validate'")

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
        "model_prediction": prediction,
        "ground_truth": None,
        "labeled_at": None,
    }
    with open(d / f"{capture_id}.json", "w") as f:
        json.dump(sidecar, f, indent=2)

    return capture_id


def record_feedback(kind, capture_id, ground_truth, base_dir=DEFAULT_CAPTURE_DIR):
    """Attaches a real-world outcome to a previously saved capture.
    Returns the updated sidecar dict, or None if the capture_id doesn't exist."""
    if kind not in ("align", "validate"):
        raise ValueError("kind must be 'align' or 'validate'")

    sidecar_path = _capture_dir(kind, base_dir) / f"{capture_id}.json"
    if not sidecar_path.exists():
        return None

    with open(sidecar_path) as f:
        sidecar = json.load(f)
    sidecar["ground_truth"] = ground_truth
    sidecar["labeled_at"] = time.time()
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    return sidecar


def load_labeled_dataset(kind, base_dir=DEFAULT_CAPTURE_DIR):
    """Every capture that has a ground_truth attached - the subset that's
    actually usable for a future retraining pass. Everything else is a
    capture still waiting to be labeled."""
    d = _capture_dir(kind, base_dir)
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
    for kind in ("align", "validate"):
        d = _capture_dir(kind, base_dir)
        total = len(list(d.glob("*.json")))
        labeled = len(load_labeled_dataset(kind, base_dir))
        stats[kind] = {"total_captured": total, "labeled": labeled, "unlabeled": total - labeled}
    return stats
