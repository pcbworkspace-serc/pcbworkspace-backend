"""
flask_server.py — PCBWorkspace SERC backend (with eager YOLO preload)
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

from pcb_jepa_nn import CLASS_NAMES, load_model, load_yolo_detector

CHECKPOINT_PATH = "best.pt"
YOLO_WEIGHTS_PATH = "yolov8_pcb.pt"
MODEL_NAME = "MobileNetV3-Small (multi-label, FPIC, paper mAP 0.636)"
HYBRID_NAME = "YOLOv8n (box proposer) + MobileNetV3-Small (classifier)"

app = Flask(__name__)
CORS(app, origins=[
    "https://pcbworkspace-serc.github.io",
    "http://localhost:8080",
    "http://localhost:5173",
])

_model = None
_model_loaded = False
_yolo = None
_device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"


def _ensure_model():
    global _model, _model_loaded, _yolo
    if _model is None and TORCH_AVAILABLE:
        _model, _model_loaded = load_model(CHECKPOINT_PATH, device=_device)
        if _model_loaded:
            print(f"  Loaded MobileNetV3-Small weights from {CHECKPOINT_PATH}")
        else:
            print(f"  WARN: {CHECKPOINT_PATH} not found — running with random init")

        # PRE-LOAD YOLO eagerly so the first /nn/detect_boxes_yolo isn't slow.
        # Adds ~5s to startup time but cuts first-request latency dramatically.
        _yolo = load_yolo_detector(_model, YOLO_WEIGHTS_PATH, eager=True)
        if _yolo.status.get("loaded"):
            print(f"  Pre-loaded YOLOv8n weights from {YOLO_WEIGHTS_PATH}")
        elif _yolo.weights_path.exists():
            print(f"  YOLO weights present but eager-load failed; will retry on first request")
        else:
            print(f"  WARN: {YOLO_WEIGHTS_PATH} missing — /nn/detect_boxes_yolo will return unavailable")
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


_ensure_model()


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "torch": TORCH_AVAILABLE,
        "pil": PIL_AVAILABLE,
        "model_loaded": _model_loaded,
        "yolo_available": _yolo.available if _yolo else False,
        "yolo_loaded": (_yolo.status.get("loaded", False) if _yolo else False),
        "trained": _model_loaded,
        "device": _device,
    })


@app.route("/nn/status")
def nn_status():
    params = sum(p.numel() for p in _model.parameters()) if _model else 0
    yolo_status = _yolo.status if _yolo else {"weights_exists": False, "loaded": False}
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
        "methods": {
            "sliding_window": {
                "endpoint": "/nn/detect_boxes",
                "available": _model_loaded,
                "description": "Fixed-grid 64px windows at scales 256/384/576, per-class NMS",
            },
            "yolo_hybrid": {
                "endpoint": "/nn/detect_boxes_yolo",
                "available": yolo_status["weights_exists"] and _model_loaded,
                "weights": yolo_status,
                "description": "YOLOv8n proposes boxes, MobileNetV3 classifies each crop",
            },
        },
    })


@app.route("/nn/detect", methods=["POST"])
def nn_detect():
    t0 = time.perf_counter()
    model = _ensure_model()
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    img = _request_to_pil()
    if img is None:
        return jsonify({
            "predictions": [], "num_classes": len(CLASS_NAMES),
            "model": MODEL_NAME, "model_kind": "classifier",
            "trained": _model_loaded, "inference_ms": 0.0,
            "error": "no image provided",
        }), 400
    result = model.infer_multilabel(img)
    result["model"] = MODEL_NAME
    result["model_kind"] = "classifier"
    result["trained"] = _model_loaded
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify(result)


@app.route("/nn/detect_boxes", methods=["POST"])
@app.route("/nn/detect_boxes_sliding", methods=["POST"])
def nn_detect_boxes():
    t0 = time.perf_counter()
    model = _ensure_model()
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    img = _request_to_pil()
    if img is None:
        return jsonify({"error": "no image provided"}), 400
    out = model.detect_boxes_sliding(img)
    out["model"] = MODEL_NAME
    out["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify(out)


@app.route("/nn/detect_boxes_yolo", methods=["POST"])
def nn_detect_boxes_yolo():
    t0 = time.perf_counter()
    model = _ensure_model()
    if model is None:
        return jsonify({"error": "Classifier not loaded"}), 503
    if _yolo is None or not _yolo.available:
        return jsonify({
            "error": "YOLO weights not available",
            "boxes": [], "n_proposals": 0,
            "method": "yolo_hybrid",
            "available": False,
            "reason": f"{YOLO_WEIGHTS_PATH} missing",
        }), 503
    img = _request_to_pil()
    if img is None:
        return jsonify({"error": "no image provided"}), 400
    try:
        out = _yolo.detect(img)
    except (FileNotFoundError, ImportError) as e:
        return jsonify({
            "error": str(e), "boxes": [], "n_proposals": 0,
            "method": "yolo_hybrid", "available": False,
        }), 503
    out["model"] = HYBRID_NAME
    out["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify(out)


@app.route("/nn/align", methods=["POST"])
def nn_align():
    return jsonify({
        "delta_theta_deg": 0.0, "delta_x_mm": 0.0, "delta_y_mm": 0.0,
        "available": False,
        "reason": "AlignmentHead not trained",
        "inference_ms": 0.0,
    })


@app.route("/nn/validate", methods=["POST"])
def nn_validate():
    return jsonify({
        "decision": "PASS", "pass_prob": 0.0, "fail_prob": 0.0,
        "available": False,
        "reason": "DefectHead not trained",
        "inference_ms": 0.0,
    })


@app.route("/nn/items", methods=["POST"])
def nn_items():
    return jsonify({"ok": True})


@app.route("/nn/items/state")
def nn_items_state():
    return jsonify({"items": [], "nn_annotations": []})


@app.route("/chat", methods=["POST"])
def chat():
    return jsonify({"reply": "[stub] keep your existing chat route here"})


if __name__ == "__main__":
    print("=" * 60)
    print(" PCBWorkspace Flask Server")
    print(f" PyTorch     : {'YES' if TORCH_AVAILABLE else 'NO'}")
    print(f" Pillow      : {'YES' if PIL_AVAILABLE else 'NO'}")
    print(f" MobileNet   : {'LOADED' if _model_loaded else 'MISSING (random init)'}")
    print(f" YOLO        : {'LOADED' if (_yolo and _yolo.status.get('loaded')) else 'MISSING'}")
    print(f" Classes     : {len(CLASS_NAMES)}")
    print(f" Device      : {_device}")
    print(f" URL         : http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
