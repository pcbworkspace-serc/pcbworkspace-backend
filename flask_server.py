"""
flask_server.py — PCBWorkspace SERC backend

Endpoints matching the frontend in pcbworkspace-v2/src/lib/nn.ts:
  /health                 - liveness probe
  /nn/status              - model info + paper metrics
  /nn/detect              - whole-image multi-label classification
  /nn/detect_boxes        - multi-scale sliding-window detection + NMS
  /nn/align               - alignment correction (stub, not trained)
  /nn/validate            - placement validation (stub, not trained)
"""

import io, time
from flask import Flask, request, jsonify

try:
    from flask_cors import CORS
except ImportError:
    class CORS:
        def __init__(self, *a, **k): pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from pcb_jepa_nn import CLASS_NAMES, load_model

CHECKPOINT_PATH = "best.pt"
MODEL_NAME = "MobileNetV3-Small (multi-label, FPIC, paper mAP 0.636)"

app = Flask(__name__)
CORS(app, origins=[
    "https://pcbworkspace-serc.github.io",
    "http://localhost:8080",
    "http://localhost:5173",
])

_model = None
_model_loaded = False
_device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"


def _ensure_model():
    global _model, _model_loaded
    if _model is None and TORCH_AVAILABLE:
        _model, _model_loaded = load_model(CHECKPOINT_PATH, device=_device)
        if _model_loaded:
            print(f"  Loaded MobileNetV3-Small weights from {CHECKPOINT_PATH}")
        else:
            print(f"  WARN: {CHECKPOINT_PATH} not found — running with random init")
    return _model


def _request_to_pil():
    if not PIL_AVAILABLE:
        return None
    raw = None
    if request.files and "image" in request.files:
        raw = request.files["image"].read()
    elif request.content_type and request.content_type.startswith("image/"):
        raw = request.get_data()
    if not raw:
        return None
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


# Load at import time so gunicorn workers get it
_ensure_model()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "torch": TORCH_AVAILABLE,
        "pil": PIL_AVAILABLE,
        "model_loaded": _model_loaded,
        "trained": _model_loaded,
        "device": _device,
    })


@app.route("/nn/status")
def nn_status():
    params = sum(p.numel() for p in _model.parameters()) if _model else 0
    return jsonify({
        "phase": "trained" if _model_loaded else "untrained",
        "loaded": _model is not None,
        "model": MODEL_NAME,
        "model_kind": "classifier",
        "parameters": params,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "device": _device,
        "trained": _model_loaded,
        "metrics_from_paper": {
            "mAP": 0.636,
            "macro_f1_at_0.5": 0.517,
            "macro_precision_at_0.5": 0.782,
            "macro_recall_at_0.5": 0.423,
        },
    })


@app.route("/nn/detect", methods=["POST"])
def nn_detect():
    """Whole-image multi-label classification (sigmoid scores)."""
    t0 = time.perf_counter()
    model = _ensure_model()
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    img = _request_to_pil()
    if img is None:
        return jsonify({
            "predictions": [],
            "num_classes": len(CLASS_NAMES),
            "model": MODEL_NAME,
            "model_kind": "classifier",
            "trained": _model_loaded,
            "inference_ms": 0.0,
            "error": "no image provided",
        }), 400
    result = model.infer_multilabel(img)
    result["model"] = MODEL_NAME
    result["model_kind"] = "classifier"
    result["trained"] = _model_loaded
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify(result)


@app.route("/nn/detect_boxes", methods=["POST"])
def nn_detect_boxes():
    """Multi-scale sliding-window detection with per-class NMS."""
    t0 = time.perf_counter()
    model = _ensure_model()
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    img = _request_to_pil()
    if img is None:
        return jsonify({"error": "no image provided"}), 400
    out = model.detect_boxes(img)
    out["model"] = MODEL_NAME
    out["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify(out)


@app.route("/nn/align", methods=["POST"])
def nn_align():
    """Not trained on this checkpoint — return unavailable."""
    return jsonify({
        "delta_theta_deg": 0.0,
        "delta_x_mm": 0.0,
        "delta_y_mm": 0.0,
        "available": False,
        "reason": "AlignmentHead not trained — only component classifier has weights",
        "inference_ms": 0.0,
    })


@app.route("/nn/validate", methods=["POST"])
def nn_validate():
    """Not trained on this checkpoint — return unavailable."""
    return jsonify({
        "decision": "PASS",
        "pass_prob": 0.0,
        "fail_prob": 0.0,
        "available": False,
        "reason": "DefectHead not trained — only component classifier has weights",
        "inference_ms": 0.0,
    })


# ── Frontend compatibility stubs ──────────────────────────────────────────────

@app.route("/nn/items", methods=["POST"])
def nn_items():
    return jsonify({"ok": True})


@app.route("/nn/items/state")
def nn_items_state():
    return jsonify({"items": [], "nn_annotations": []})


@app.route("/chat", methods=["POST"])
def chat():
    return jsonify({"reply": "[stub] keep your existing chat route here"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print(" PCBWorkspace Flask Server — MobileNetV3-Small (FPIC)")
    print(f" PyTorch  : {'YES' if TORCH_AVAILABLE else 'NO'}")
    print(f" Pillow   : {'YES' if PIL_AVAILABLE else 'NO'}")
    print(f" Weights  : {'LOADED' if _model_loaded else 'MISSING (random init)'}")
    print(f" Classes  : {len(CLASS_NAMES)}")
    print(f" Device   : {_device}")
    print(f" URL      : http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
