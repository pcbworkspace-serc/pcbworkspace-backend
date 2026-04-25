import sys, time, threading, io
from pathlib import Path
from flask import Flask, request, jsonify

try:
    from flask_cors import CORS
except ImportError:
    class CORS:
        def __init__(self, app): pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    import torchvision.transforms as T
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))

try:
    from pcb_jepa_nn import JEPAConfig, PCBVisionSystem, multitask_loss
    JEPA_AVAILABLE = True
except ImportError:
    JEPA_AVAILABLE = False

app = Flask(__name__)
CORS(app)

CHECKPOINT_PATH = "jepa_checkpoint.pt"
COMPONENT_CLASSES = [
    "Resistor","Capacitor","Diode","LED","Transistor",
    "Channel Port","IC","Crystal","Inductor","Fuse",
    "Button","Connector","SOT-23","QFP","BGA",
    "0402","0603","0805","1206","SOD-123"
]

_model       = None
_cfg         = None
_model_phase = "untrained"
_board_items = []
_lock        = threading.Lock()
_training    = {
    "running":False,"phase":"idle","epoch":0,"total_epochs":0,
    "loss":0.0,"l_comp":0.0,"l_bbox":0.0,"l_defect":0.0,
    "comp_acc":0.0,"defect_acc":0.0,
    "elapsed_seconds":0.0,"eta_seconds":0.0,
}

# ── Image preprocessing ───────────────────────────────────────────────────────

_transform = None

def get_transform():
    global _transform
    if _transform is None and PIL_AVAILABLE:
        _transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    return _transform


def request_to_tensor(req):
    """Decode image from multipart or raw binary request body."""
    if not (PIL_AVAILABLE and TORCH_AVAILABLE):
        return None
    raw = None
    if req.files and "image" in req.files:
        raw = req.files["image"].read()
    elif req.content_type and req.content_type.startswith("image/"):
        raw = req.get_data()
    if raw is None:
        return None
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return get_transform()(img).unsqueeze(0)
    except Exception:
        return None


def dummy_tensor():
    if TORCH_AVAILABLE:
        return torch.zeros(1, 3, 224, 224)
    return None

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    global _model, _cfg, _model_phase
    if not JEPA_AVAILABLE:
        return
    _cfg   = JEPAConfig()
    if TORCH_AVAILABLE:
        _model = PCBVisionSystem(_cfg)
        if Path(CHECKPOINT_PATH).exists():
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
            _model.load_state_dict(ckpt["model_state"])
            _model_phase = ckpt.get("phase", "trained")
            print(f"  Loaded checkpoint (phase={_model_phase}, "
                  f"val_loss={ckpt.get('val_loss','?'):.4f}, "
                  f"comp_acc={ckpt.get('comp_acc','?'):.1f}%)")
        else:
            print("  No checkpoint found — run train.py first for accurate results")
        _model.eval()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "ok":     True,
        "torch":  TORCH_AVAILABLE,
        "opencv": CV2_AVAILABLE,
        "jepa":   JEPA_AVAILABLE,
        "pil":    PIL_AVAILABLE,
        "trained": Path(CHECKPOINT_PATH).exists(),
    })


@app.route("/nn/status")
def nn_status():
    params = (sum(p.numel() for p in _model.parameters())
              if _model and TORCH_AVAILABLE else 0)
    return jsonify({
        "loaded":     _model is not None,
        "model":      "PCBVisionSystem (CNN)",
        "parameters": params,
        "device":     "cpu",
        "checkpoint": CHECKPOINT_PATH if Path(CHECKPOINT_PATH).exists() else None,
        "phase":      _model_phase,
        "heads":      ["ComponentHead", "DefectHead", "AlignmentHead"],
    })


@app.route("/nn/detect", methods=["POST"])
def nn_detect():
    t0 = time.perf_counter()
    if _model is None or not TORCH_AVAILABLE:
        return jsonify({"error": "Model not loaded"}), 503
    t = request_to_tensor(request); x = t if t is not None else dummy_tensor()
    result = _model.infer_detect(x)
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify(result)


@app.route("/nn/align", methods=["POST"])
def nn_align():
    t0 = time.perf_counter()
    if _model is None or not TORCH_AVAILABLE:
        return jsonify({"error": "Model not loaded"}), 503
    t = request_to_tensor(request); x = t if t is not None else dummy_tensor()
    result = _model.infer_align(x)
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify(result)


@app.route("/nn/validate", methods=["POST"])
def nn_validate():
    t0 = time.perf_counter()
    if _model is None or not TORCH_AVAILABLE:
        return jsonify({"error": "Model not loaded"}), 503
    t = request_to_tensor(request); x = t if t is not None else dummy_tensor()
    result = _model.infer_validate(x)
    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify(result)


@app.route("/nn/items", methods=["POST"])
def nn_items():
    global _board_items
    _board_items = request.get_json().get("items", [])
    return jsonify({"ok": True, "item_count": len(_board_items)})


@app.route("/nn/items/state")
def nn_items_state():
    return jsonify({"items": _board_items, "nn_annotations": []})


@app.route("/nn/train/start", methods=["POST"])
def nn_train_start():
    """Kick off training in a background thread."""
    data       = request.get_json() or {}
    epochs     = int(data.get("epochs", 50))
    batch_size = int(data.get("batch_size", 16))
    lr         = float(data.get("lr", 1e-3))

    def _run():
        import subprocess, sys
        with _lock:
            _training.update({"running": True, "phase": "training",
                               "epoch": 0, "total_epochs": epochs})
        result = subprocess.run(
            [sys.executable, "train.py",
             "--epochs", str(epochs),
             "--batch-size", str(batch_size),
             "--lr", str(lr)],
            capture_output=True, text=True
        )
        # Reload model after training
        load_model()
        with _lock:
            _training.update({"running": False, "phase": "done"})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"started": True,
                    "message": f"Training started: {epochs} epochs. Check terminal for progress."})


@app.route("/nn/train/status")
def nn_train_status():
    with _lock:
        return jsonify(dict(_training))


@app.route("/chat", methods=["POST"])
def chat():
    return jsonify({"reply": "[stub] keep your existing chat route here"})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  PCBWorkspace Flask Server — CNN Multi-Task")
    print(f"  PyTorch  : {'YES' if TORCH_AVAILABLE else 'NO'}")
    print(f"  OpenCV   : {'YES' if CV2_AVAILABLE  else 'NO'}")
    print(f"  Pillow   : {'YES' if PIL_AVAILABLE  else 'NO'}")
    print(f"  CNN Model: {'YES' if JEPA_AVAILABLE else 'NO — pcb_jepa_nn.py missing'}")
    print(f"  Trained  : {'YES' if Path(CHECKPOINT_PATH).exists() else 'NO — run: python train.py'}")
    print("  URL      : http://127.0.0.1:5000")
    print("=" * 55)
    load_model()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
