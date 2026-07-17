"""
flask_server.py — PCBWorkspace SERC backend (with eager YOLO preload + calibration)
"""

import io, time, json, os, threading, math
from pathlib import Path
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

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from pcb_jepa_nn import CLASS_NAMES, load_model, load_yolo_detector
import layout_engine
import kicad_export
import vision_classical
import jepa_heads
import data_capture

CAPTURE_TRAINING_DATA = os.environ.get("CAPTURE_TRAINING_DATA", "").lower() in ("1", "true", "yes")
CAPTURE_DIR = os.environ.get("CAPTURE_DIR", data_capture.DEFAULT_CAPTURE_DIR)

CHECKPOINT_PATH = "best.pt"
YOLO_WEIGHTS_PATH = "yolov8_pcb.pt"
MODEL_NAME = "MobileNetV3-Small (multi-label, FPIC, paper mAP 0.636)"
HYBRID_NAME = "YOLOv8n (box proposer) + MobileNetV3-Small (classifier)"

CALIBRATION_PATH = "calibration.json"
DEFAULT_BOARD_W_MM = 62.0
DEFAULT_BOARD_H_MM = 42.0
_calibration_lock = threading.Lock()


def _load_saved_calibration():
    try:
        with open(CALIBRATION_PATH, "r") as f:
            data = json.load(f)
            return data.get("homography"), data.get("board_size_mm")
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None


_saved_homography, _saved_board_size = _load_saved_calibration()


def _get_board_bounds():
    """Board size used by /vla/plan and /layout/*. Prefers the size saved
    alongside the live calibration (so a real board's calibration and the
    robot's motion planner agree) and only falls back to the historical
    62x42mm default if no calibration has been saved yet."""
    with _calibration_lock:
        size = _saved_board_size
    if size and "w_mm" in size and "h_mm" in size:
        return float(size["w_mm"]), float(size["h_mm"])
    return DEFAULT_BOARD_W_MM, DEFAULT_BOARD_H_MM


def _estimate_px_per_mm(homography):
    """Local-Jacobian estimate of pixel-to-mm scale around the image center,
    from the saved perspective homography. Not exact off-center on a
    strongly non-fronto-parallel camera, but good enough for converting a
    classical-CV pixel offset into an approximate mm correction."""
    if not (CV2_AVAILABLE and homography):
        return None
    try:
        H = np.array(homography, dtype=np.float64)
        p0 = np.array([[[500.0, 500.0]]], dtype=np.float64)
        p1 = np.array([[[510.0, 500.0]]], dtype=np.float64)
        w0 = cv2.perspectiveTransform(p0, H)[0][0]
        w1 = cv2.perspectiveTransform(p1, H)[0][0]
        mm_dist = float(np.hypot(w1[0] - w0[0], w1[1] - w0[1]))
        if mm_dist <= 1e-6:
            return None
        return 10.0 / mm_dist  # px per mm
    except Exception:
        return None


def _extract_json_block(raw):
    """Strip an optional ```json ... ``` fence Claude sometimes wraps
    structured output in, then json.loads it."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


def _claude_complete(system, messages, max_tokens=1024):
    """Shared Claude call used by /chat, /vla/plan and /schematic/generate —
    previously each endpoint built its own anthropic.Anthropic() client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set on server")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return msg.content[0].text.strip()


app = Flask(__name__)
CORS(app, origins=[
    "https://pcbworkspace-serc.github.io",
    "http://localhost:8080",
    "http://localhost:5173",
])

# Basic request logging — one line per request to stdout, which Render
# (and any platform running `python flask_server.py` under a process
# manager) captures as logs automatically. No new dependency, no file
# handling to get wrong on a read-only filesystem.
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
_request_logger = logging.getLogger("pcbworkspace.request")


@app.before_request
def _log_request_start():
    request._t0 = time.perf_counter()


@app.after_request
def _log_request_end(response):
    duration_ms = round((time.perf_counter() - getattr(request, "_t0", time.perf_counter())) * 1000, 1)
    _request_logger.info(
        "%s %s -> %d (%sms)%s",
        request.method, request.path, response.status_code, duration_ms,
        " key=missing" if response.status_code == 401 else "",
    )
    return response

# API key auth — OFF by default (BACKEND_API_KEY unset) so this doesn't
# break the deployed frontend the moment it ships; every route below is
# reachable with no auth today. Set BACKEND_API_KEY on Render to turn it
# on, and update the frontend to send it as X-API-Key on every request
# first — flipping this on without that change locks your own site out.
import functools

def require_api_key(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        configured_key = os.environ.get("BACKEND_API_KEY")
        if not configured_key:
            return fn(*args, **kwargs)  # auth disabled - not configured
        if request.method == "OPTIONS":
            return fn(*args, **kwargs)  # let CORS preflight through unauthenticated
        sent_key = request.headers.get("X-API-Key", "")
        if sent_key != configured_key:
            return jsonify({"error": "missing or invalid X-API-Key header"}), 401
        return fn(*args, **kwargs)
    return wrapper


_model = None
_model_loaded = False
_yolo = None
_device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"

_align_cnn = None
_align_head = None
_validate_head = None
_jepa_heads_meta = None
_jepa_heads_load_attempted = False


def _ensure_model():
    global _model, _model_loaded, _yolo
    if _model is None and TORCH_AVAILABLE:
        _model, _model_loaded = load_model(CHECKPOINT_PATH, device=_device)
        if _model_loaded:
            print(f"  Loaded MobileNetV3-Small weights from {CHECKPOINT_PATH}")
        else:
            print(f"  WARN: {CHECKPOINT_PATH} not found - running with random init")

        _yolo = load_yolo_detector(_model, YOLO_WEIGHTS_PATH, eager=True)
        if _yolo.status.get("loaded"):
            print(f"  Pre-loaded YOLOv8n weights from {YOLO_WEIGHTS_PATH}")
        elif _yolo.weights_path.exists():
            print(f"  YOLO weights present but eager-load failed; will retry on first request")
        else:
            print(f"  WARN: {YOLO_WEIGHTS_PATH} missing - /nn/detect_boxes_yolo will return unavailable")
    return _model


def _ensure_jepa_heads():
    """Loads jepa_heads_synthetic.pt once, if present. These heads are
    trained on synthetic domain-randomized data only (see jepa_heads.py) —
    real, validated numbers on synthetic holdout data, not on your actual
    camera feed. /nn/align and /nn/validate use them as the primary path
    when available and fall back to vision_classical.py's classical-CV
    heuristics otherwise, so a missing/corrupt checkpoint degrades instead
    of breaking the endpoint."""
    global _align_cnn, _align_head, _validate_head, _jepa_heads_meta, _jepa_heads_load_attempted
    if _jepa_heads_load_attempted:
        return _align_cnn, _align_head, _validate_head
    _jepa_heads_load_attempted = True
    if not TORCH_AVAILABLE:
        return None, None, None
    align_cnn, align_head, validate_head, meta = jepa_heads.load_heads()
    if align_cnn is not None:
        print(f"  Loaded trained align/validate heads from {jepa_heads.CHECKPOINT_PATH}"
              f" (synthetic val: angle_MAE={meta['align_metrics']['val_angle_mae_deg']}deg,"
              f" validate_acc={meta['validate_metrics']['val_accuracy']})")
    else:
        print(f"  {jepa_heads.CHECKPOINT_PATH} not found - /nn/align and /nn/validate use classical-CV fallback."
              f" Run `python jepa_heads.py` to train it.")
    _align_cnn, _align_head, _validate_head, _jepa_heads_meta = align_cnn, align_head, validate_head, meta
    return _align_cnn, _align_head, _validate_head


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
_ensure_jepa_heads()


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "torch": TORCH_AVAILABLE,
        "pil": PIL_AVAILABLE,
        "opencv": CV2_AVAILABLE,
        "model_loaded": _model_loaded,
        "yolo_available": _yolo.available if _yolo else False,
        "yolo_loaded": (_yolo.status.get("loaded", False) if _yolo else False),
        "trained": _model_loaded,
        "device": _device,
        "calibrated": _saved_homography is not None,
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
            "align": {
                "endpoint": "/nn/align",
                "available": CV2_AVAILABLE or _align_cnn is not None,
                "method": "trained_head_synthetic (primary)" if _align_cnn is not None else "classical_cv_min_area_rect (fallback, no trained checkpoint present)",
                "synthetic_val_metrics": (_jepa_heads_meta or {}).get("align_metrics"),
                "description": "AlignCNN+AlignHead trained on synthetic domain-randomized data (jepa_heads.py) if jepa_heads_synthetic.pt is present, else classical-CV contour/minAreaRect fallback. Not yet validated against real camera captures either way.",
            },
            "validate": {
                "endpoint": "/nn/validate",
                "available": True,
                "method": "trained_head_synthetic (default) or classifier_confidence (with expected_class) or classical_cv_presence (final fallback)",
                "synthetic_val_metrics": (_jepa_heads_meta or {}).get("validate_metrics"),
                "description": "ValidateHead trained on synthetic domain-randomized data if jepa_heads_synthetic.pt is present, else classical-CV/classifier fallback.",
            },
        },
        "schematic_and_layout": {
            "generate": {"endpoint": "/schematic/generate", "description": "NL description -> structured netlist JSON + DRC/ERC + BOM plausibility check"},
            "drc": {"endpoint": "/schematic/drc", "description": "Run DRC/ERC on an existing circuit JSON without calling Claude again"},
            "bom_check": {"endpoint": "/schematic/bom_check", "description": "Deterministic value/footprint sanity check - stand-in for live Octopart/Digikey sourcing (no API key available for that)"},
            "export_kicad": {"endpoint": "/schematic/export/kicad", "description": "Circuit JSON -> KiCad legacy netlist file, importable via pcbnew's Read Netlist"},
            "place": {"endpoint": "/layout/place", "description": "Simulated-annealing auto-placement (HPWL + overlap cost)"},
            "route": {"endpoint": "/layout/route", "description": "Pin-to-pin Lee/BFS auto-router across 1-2 layers, resolving crossings with a layer hop where possible - see layer_note in response"},
        },
        "robot_safety": {
            "board_bounds_clamp": {"description": "clamp_vla_action - drops/clamps out-of-range or malformed robot actions"},
            "kinematic_reach_check": {"description": "filter_unreachable_actions - drops in-bounds-but-unreachable targets (ARM_BASE_OFFSET_MM is an unmeasured placeholder, see layout_engine.py)"},
        },
        "data_capture": {
            "enabled": CAPTURE_TRAINING_DATA,
            "feedback_endpoint": "/nn/feedback",
            "stats_endpoint": "/nn/capture_stats",
        },
    })


@app.route("/nn/detect", methods=["POST"])
@require_api_key
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
@require_api_key
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
@require_api_key
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


def _run_trained_align(img, align_cnn, align_head, px_per_mm):
    resized = img.resize((jepa_heads.WIN_SIZE, jepa_heads.WIN_SIZE))
    x = jepa_heads._to_tensor(resized).unsqueeze(0)
    with torch.no_grad():
        out = align_head(align_cnn(x))[0]
    theta = math.degrees(math.atan2(float(out[0]), float(out[1])))
    dx_px = float(out[2]) * jepa_heads.MAX_OFFSET_PX
    dy_px = float(out[3]) * jepa_heads.MAX_OFFSET_PX
    result = {
        "delta_theta_deg": round(theta, 2),
        "delta_x_px": round(dx_px, 2),
        "delta_y_px": round(dy_px, 2),
        "available": True,
        "method": "trained_head_synthetic",
        "note": "trained on synthetic domain-randomized data only - not yet validated against real camera captures",
        "synthetic_val_metrics": _jepa_heads_meta["align_metrics"] if _jepa_heads_meta else None,
    }
    if px_per_mm:
        result["delta_x_mm"] = round(dx_px / px_per_mm, 3)
        result["delta_y_mm"] = round(dy_px / px_per_mm, 3)
    else:
        result["delta_x_mm"] = None
        result["delta_y_mm"] = None
    return result


@app.route("/nn/align", methods=["POST"])
@require_api_key
def nn_align():
    t0 = time.perf_counter()
    img = _request_to_pil()
    if img is None:
        return jsonify({
            "delta_theta_deg": 0.0, "delta_x_mm": 0.0, "delta_y_mm": 0.0,
            "available": False, "reason": "no image provided", "inference_ms": 0.0,
        }), 400
    with _calibration_lock:
        px_per_mm = _estimate_px_per_mm(_saved_homography)

    align_cnn, align_head, _ = _ensure_jepa_heads()
    if align_cnn is not None:
        try:
            result = _run_trained_align(img, align_cnn, align_head, px_per_mm)
        except Exception as e:
            result = vision_classical.estimate_alignment(img, px_per_mm=px_per_mm)
            result["fallback_reason"] = "trained head inference failed: " + str(e)
    else:
        result = vision_classical.estimate_alignment(img, px_per_mm=px_per_mm)

    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if CAPTURE_TRAINING_DATA:
        result["capture_id"] = data_capture.save_capture("align", img, result, base_dir=CAPTURE_DIR)
    return jsonify(result)


@app.route("/nn/validate", methods=["POST"])
@require_api_key
def nn_validate():
    t0 = time.perf_counter()
    img = _request_to_pil()
    if img is None:
        return jsonify({
            "decision": "FAIL", "pass_prob": 0.0, "fail_prob": 0.0,
            "available": False, "reason": "no image provided", "inference_ms": 0.0,
        }), 400
    expected_class = (request.form.get("expected_class") or request.args.get("expected_class") or "").strip() or None

    if expected_class:
        # a specific expected class was named - that's a more direct check
        # than the general trained head answers, so use the classifier path
        model = _ensure_model()
        result = vision_classical.validate_placement(
            img, model=model, class_names=CLASS_NAMES, expected_class=expected_class,
        )
    else:
        _, _, validate_head = _ensure_jepa_heads()
        if validate_head is not None:
            try:
                model = _ensure_model()
                fx = jepa_heads.FrozenFeatureExtractor(model.backbone)
                resized = img.resize((jepa_heads.WIN_SIZE, jepa_heads.WIN_SIZE))
                x = jepa_heads._to_tensor(resized).unsqueeze(0)
                with torch.no_grad():
                    logits = validate_head(fx(x))[0]
                    probs = torch.softmax(logits, dim=-1)
                fail_prob, pass_prob = float(probs[0]), float(probs[1])
                result = {
                    "decision": "PASS" if pass_prob >= fail_prob else "FAIL",
                    "pass_prob": round(pass_prob, 3), "fail_prob": round(fail_prob, 3),
                    "available": True, "method": "trained_head_synthetic",
                    "note": "trained on synthetic domain-randomized data only - not yet validated against real camera captures",
                    "synthetic_val_metrics": _jepa_heads_meta["validate_metrics"] if _jepa_heads_meta else None,
                }
            except Exception as e:
                result = vision_classical.validate_placement(img)
                result["fallback_reason"] = "trained head inference failed: " + str(e)
        else:
            result = vision_classical.validate_placement(img)

    result["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    if CAPTURE_TRAINING_DATA:
        result["capture_id"] = data_capture.save_capture("validate", img, result, base_dir=CAPTURE_DIR)
    return jsonify(result)


@app.route("/nn/feedback", methods=["POST"])
@require_api_key
def nn_feedback():
    """Attaches a real-world outcome to a capture saved by /nn/align or
    /nn/validate (when CAPTURE_TRAINING_DATA is on). This is the second
    half of the data-collection loop: the model's own prediction isn't
    training data, but prediction + a later-confirmed ground truth is —
    see data_capture.py."""
    data = request.get_json(silent=True) or {}
    kind = data.get("kind")
    capture_id = data.get("capture_id")
    ground_truth = data.get("ground_truth")
    if kind not in ("align", "validate") or not capture_id or ground_truth is None:
        return jsonify({"ok": False, "error": "kind ('align'|'validate'), capture_id, and ground_truth are required"}), 400

    updated = data_capture.record_feedback(kind, capture_id, ground_truth, base_dir=CAPTURE_DIR)
    if updated is None:
        return jsonify({"ok": False, "error": "no capture found for that kind/capture_id"}), 404
    return jsonify({"ok": True, "capture": updated})


@app.route("/nn/capture_stats")
def nn_capture_stats():
    return jsonify({
        "capture_enabled": CAPTURE_TRAINING_DATA,
        "stats": data_capture.capture_stats(base_dir=CAPTURE_DIR),
    })


@app.route("/nn/items", methods=["POST"])
@require_api_key
def nn_items():
    return jsonify({"ok": True})


@app.route("/nn/items/state")
def nn_items_state():
    return jsonify({"items": [], "nn_annotations": []})


# Calibration endpoints
@app.route("/calibration/save", methods=["POST"])
@require_api_key
def calibration_save():
    global _saved_homography, _saved_board_size
    if not CV2_AVAILABLE:
        return jsonify({"error": "OpenCV not installed"}), 503

    data = request.get_json() or {}
    pixel_pts = data.get("pixel_points")
    world_pts = data.get("world_points")
    # optional: the real physical board size (mm) this calibration was taken
    # against. When present, /vla/plan and /layout/* use it instead of the
    # hardcoded 62x42mm default, so vision calibration and motion planning
    # stay in sync for whatever board is actually on the bed.
    board_size_mm = data.get("board_size_mm")
    if board_size_mm is not None:
        try:
            board_size_mm = {"w_mm": float(board_size_mm["w_mm"]), "h_mm": float(board_size_mm["h_mm"])}
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "board_size_mm must be {w_mm, h_mm} if provided"}), 400

    if not pixel_pts or not world_pts or len(pixel_pts) != len(world_pts):
        return jsonify({"error": "Need matching pixel_points and world_points"}), 400
    if len(pixel_pts) < 4:
        return jsonify({"error": "Need at least 4 point pairs"}), 400

    try:
        src = np.array(pixel_pts, dtype=np.float32)
        dst = np.array(world_pts, dtype=np.float32)
        H, _mask = cv2.findHomography(src, dst, method=0)
        if H is None:
            return jsonify({"error": "Homography computation failed (collinear points?)"}), 400

        H_list = H.tolist()
        with _calibration_lock:
            _saved_homography = H_list
            _saved_board_size = board_size_mm
            try:
                with open(CALIBRATION_PATH, "w") as f:
                    json.dump({
                        "homography": H_list,
                        "point_count": len(pixel_pts),
                        "board_size_mm": board_size_mm,
                    }, f)
            except OSError as e:
                print(f"  WARN: could not persist calibration to disk: {e}")

        return jsonify({"ok": True, "homography": H_list, "board_size_mm": board_size_mm})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/calibration/get")
def calibration_get():
    with _calibration_lock:
        H = _saved_homography
        board_size_mm = _saved_board_size
    if H is None:
        return jsonify({"error": "No calibration saved"}), 404
    return jsonify({"homography": H, "board_size_mm": board_size_mm})


@app.route("/calibration/clear", methods=["POST"])
@require_api_key
def calibration_clear():
    global _saved_homography, _saved_board_size
    with _calibration_lock:
        _saved_homography = None
        _saved_board_size = None
        try:
            os.remove(CALIBRATION_PATH)
        except (FileNotFoundError, OSError):
            pass
    return jsonify({"ok": True})


CHAT_SYSTEM = "You are Layla, an expert PCB design and electrical-engineering assistant built into SERC's PCBWorkspace. You help users design circuits, choose components, lay out boards, understand protocols, and plan robot assembly. You are knowledgeable, friendly, and concise. When a user asks what they can do, explain the workspace's capabilities: designing PCBs, placing components, driving the MiniMEE robot arm, and running vision detection. Answer engineering questions directly and practically. Keep replies focused - a few sentences to a few short paragraphs. Use plain language."

@app.route("/chat", methods=["POST", "OPTIONS"])
@require_api_key
def chat():
    if request.method == "OPTIONS":
        return "", 200
    data = request.get_json(silent=True) or {}
    history = data.get("messages")
    if not history:
        single = (data.get("message") or "").strip()
        history = [{"role": "user", "content": single}] if single else []
    clean = [m for m in history
             if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()]
    while clean and clean[0]["role"] != "user":
        clean.pop(0)
    if not clean:
        return jsonify({"reply": "What would you like to work on?"})

    try:
        reply = _claude_complete(CHAT_SYSTEM, clean, max_tokens=1024)
        return jsonify({"reply": reply})
    except RuntimeError as e:
        return jsonify({"reply": "(Layla backend missing ANTHROPIC_API_KEY)"}), 500
    except Exception as e:
        return jsonify({"reply": "Error: " + str(e)}), 500


# VLA plan (Layla natural-language to robot actions)
VLA_Z_MAX_MM = 20.0

def _vla_system_prompt(board_w, board_h):
    cx, cy = round(board_w / 2, 1), round(board_h / 2, 1)
    return f"""You are Layla, a robot arm controller for a PCB assembly robot (MiniMEE by SERC).
Convert natural language instructions into a sequence of robot actions.

Board: {board_w:g} x {board_h:g} mm. Origin (0,0) is bottom-left. Center is ({cx}, {cy}) mm.

Respond ONLY with valid JSON - no markdown, no explanation, just raw JSON:
{{
  "interpretation": "one-line description of what you will do",
  "actions": [
    {{"action": "home"}},
    {{"action": "move", "x_mm": {cx}, "y_mm": {cy}, "z_mm": 5}},
    {{"action": "pick"}},
    {{"action": "move", "x_mm": {cx}, "y_mm": {cy}, "z_mm": 0}},
    {{"action": "place"}}
  ],
  "warnings": []
}}

Valid action types:
  move - requires x_mm (0-{board_w:g}), y_mm (0-{board_h:g}), z_mm (0-{VLA_Z_MAX_MM:g})
  rotate - requires degrees
  home | pick | place | release | scan | detect | align | validate - no extra fields

Rules:
- z_mm = 5 for transit moves, z_mm = 0 for pick/place
- Always HOME first unless told not to
- If the instruction has no robot motion intent, return an empty actions array
- Keep x_mm within 0-{board_w:g}, y_mm within 0-{board_h:g}

Note: whatever coordinates you return here are re-validated and clamped
server-side against the same board bounds before anything reaches a motor —
stay inside them, but this is a safety net, not a substitute for it."""

@app.route("/vla/plan", methods=["POST", "OPTIONS"])
@require_api_key
def vla_plan():
    if request.method == "OPTIONS":
        return "", 200

    instruction = request.form.get("instruction", "").strip()
    board_state_raw = request.form.get("board_state", "[]")
    try:
        board_state = json.loads(board_state_raw)
    except Exception:
        board_state = []

    board_w, board_h = _get_board_bounds()

    try:
        board_summary = json.dumps(board_state[:10])
        user_msg = "Current board state (up to 10 components):\n" + board_summary + "\n\nInstruction: " + instruction
        raw = _claude_complete(_vla_system_prompt(board_w, board_h),
                                [{"role": "user", "content": user_msg}], max_tokens=1024)
        result = _extract_json_block(raw)

        # Server-side safety net: never trust LLM-returned coordinates to be
        # in range just because the prompt asked nicely. Anything malformed
        # or wildly out of the board envelope is dropped here, not clamped
        # blindly and sent to a real motor.
        raw_actions = result.get("actions", [])
        clean_actions, clamp_warnings = layout_engine.clamp_vla_plan(
            raw_actions, board_w, board_h, z_max=VLA_Z_MAX_MM,
        )
        # second safety net: a target can be inside the board rectangle and
        # still be outside what the arm's own 2-link kinematics can reach
        clean_actions, reach_warnings = layout_engine.filter_unreachable_actions(clean_actions)
        clamp_warnings = clamp_warnings + reach_warnings

        return jsonify({
            "ok": True,
            "actions": clean_actions,
            "interpretation": result.get("interpretation", instruction),
            "warnings": result.get("warnings", []) + clamp_warnings,
            "board_size_mm": {"w_mm": board_w, "h_mm": board_h},
            "dropped_action_count": len(raw_actions) - len(clean_actions),
        })
    except json.JSONDecodeError as e:
        return jsonify({"ok": False, "error": "Could not parse Claude response: " + str(e),
                        "actions": [], "interpretation": ""}), 500
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e), "actions": [], "interpretation": ""}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e),
                        "actions": [], "interpretation": ""}), 500


# Structured schematic generation + DRC/ERC
# ---------------------------------------------------------------------------
# /chat only ever returns prose — there's no structured circuit object for
# PCB Workspace's canvas to render, and nothing to run a rule-check against
# (the "connectivity hallucination" problem: an LLM can describe a circuit
# that doesn't actually connect the way it claims). This endpoint asks
# Claude for a strict JSON netlist instead of prose, then runs it through
# layout_engine.run_drc_erc() so a hallucinated connection is caught
# mechanically rather than trusted.
SCHEMATIC_SYSTEM = """You are Layla's schematic generator for SERC's PCBWorkspace.
Convert a natural-language circuit description into a structured netlist.

Respond ONLY with valid JSON - no markdown, no explanation:
{
  "interpretation": "one-line description of the circuit",
  "circuit": {
    "components": [
      {"ref": "R1", "type": "resistor", "value": "10k", "footprint": "0603",
       "pins": {"1": "VIN", "2": "MID"}},
      {"ref": "R2", "type": "resistor", "value": "10k", "footprint": "0603",
       "pins": {"1": "MID", "2": "GND"}}
    ]
  }
}

Rules:
- Every component needs a unique "ref" (R1, C1, U1, D1, Q1, J1, ...)
- "type" is one of: resistor, capacitor, inductor, diode, led, crystal, fuse,
  switch, transistor, mosfet, regulator, connector, ic
- Every pin must map to a net name string - never leave a pin unassigned
- Include a connector or source component for any net that's meant to enter
  or leave the circuit (e.g. a J1 power connector for VIN/GND) so those
  nets aren't left with only one connection
- Use "GND" for ground unless the user names it differently
- footprint is one of: 0402, 0603, 0805, sot-23, sot-223, soic-8, qfp-32,
  dip-8 (best guess for the component type/value if not specified)"""


@app.route("/schematic/generate", methods=["POST", "OPTIONS"])
@require_api_key
def schematic_generate():
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True) or {}
    description = (data.get("description") or "").strip()
    if not description:
        return jsonify({"ok": False, "error": "description is required"}), 400

    try:
        raw = _claude_complete(SCHEMATIC_SYSTEM,
                                [{"role": "user", "content": description}], max_tokens=2048)
        result = _extract_json_block(raw)
        circuit = result.get("circuit", {})
        drc = layout_engine.run_drc_erc(circuit)
        bom_check = layout_engine.check_bom_plausibility(circuit)
        return jsonify({
            "ok": True,
            "interpretation": result.get("interpretation", description),
            "circuit": circuit,
            "drc": drc,
            "bom_check": bom_check,
        })
    except json.JSONDecodeError as e:
        return jsonify({"ok": False, "error": "Could not parse Claude response: " + str(e)}), 500
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/schematic/drc", methods=["POST"])
@require_api_key
def schematic_drc():
    """Run DRC/ERC on a circuit the frontend already has (e.g. after a user
    hand-edits a generated netlist), without calling Claude again."""
    data = request.get_json(silent=True) or {}
    circuit = data.get("circuit")
    if not circuit:
        return jsonify({"ok": False, "error": "circuit is required"}), 400
    return jsonify({"ok": True, "drc": layout_engine.run_drc_erc(circuit)})


@app.route("/schematic/bom_check", methods=["POST"])
@require_api_key
def schematic_bom_check():
    """Deterministic value/footprint sanity check — a stand-in for live
    Octopart/Digikey sourcing (no API key available for that here). See
    layout_engine.check_bom_plausibility's docstring."""
    data = request.get_json(silent=True) or {}
    circuit = data.get("circuit")
    if not circuit:
        return jsonify({"ok": False, "error": "circuit is required"}), 400
    return jsonify({"ok": True, "bom_check": layout_engine.check_bom_plausibility(circuit)})


@app.route("/schematic/export/kicad", methods=["POST"])
@require_api_key
def schematic_export_kicad():
    """circuit -> KiCad legacy netlist file (importable via pcbnew's "Read
    Netlist"). This has not been round-tripped through a real KiCad
    install in this environment - see kicad_export.py's docstring."""
    data = request.get_json(silent=True) or {}
    circuit = data.get("circuit")
    if not circuit:
        return jsonify({"ok": False, "error": "circuit is required"}), 400
    netlist_text = kicad_export.to_kicad_netlist(circuit)
    from flask import Response
    return Response(
        netlist_text, mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=layla_export.net"},
    )


# Auto-placement / auto-routing
# ---------------------------------------------------------------------------
@app.route("/layout/place", methods=["POST"])
@require_api_key
def layout_place():
    data = request.get_json(silent=True) or {}
    components = data.get("components")
    if not components:
        return jsonify({"ok": False, "error": "components is required"}), 400

    default_w, default_h = _get_board_bounds()
    board_w = float(data.get("board_w_mm") or default_w)
    board_h = float(data.get("board_h_mm") or default_h)
    iterations = int(data.get("iterations") or 3000)
    iterations = max(200, min(iterations, 20000))  # keep request latency bounded
    seed = data.get("seed")

    result = layout_engine.auto_place(components, board_w, board_h, iterations=iterations, seed=seed)
    result["ok"] = True
    result["board_size_mm"] = {"w_mm": board_w, "h_mm": board_h}
    return jsonify(result)


@app.route("/layout/route", methods=["POST"])
@require_api_key
def layout_route():
    data = request.get_json(silent=True) or {}
    components = data.get("components")
    if not components:
        return jsonify({"ok": False, "error": "components is required"}), 400

    default_w, default_h = _get_board_bounds()
    board_w = float(data.get("board_w_mm") or default_w)
    board_h = float(data.get("board_h_mm") or default_h)
    grid_mm = float(data.get("grid_mm") or 1.0)
    layers = int(data.get("layers") or 2)

    positions = data.get("positions")
    placement_info = None
    if not positions:
        # no placement supplied - run one so /layout/route works standalone
        placement_info = layout_engine.auto_place(components, board_w, board_h,
                                                    iterations=int(data.get("iterations") or 3000),
                                                    seed=data.get("seed"))
        positions = placement_info["positions"]

    result = layout_engine.auto_route(positions, components, board_w, board_h, grid_mm=grid_mm, layers=layers)
    result["ok"] = True
    result["board_size_mm"] = {"w_mm": board_w, "h_mm": board_h}
    result["positions_used"] = positions
    if placement_info is not None:
        result["placement"] = {"cost": placement_info["cost"], "iterations": placement_info["iterations"]}
    return jsonify(result)


if __name__ == "__main__":
    print("=" * 60)
    print(" PCBWorkspace Flask Server")
    print(f" PyTorch     : {'YES' if TORCH_AVAILABLE else 'NO'}")
    print(f" Pillow      : {'YES' if PIL_AVAILABLE else 'NO'}")
    print(f" OpenCV      : {'YES' if CV2_AVAILABLE else 'NO'}")
    print(f" MobileNet   : {'LOADED' if _model_loaded else 'MISSING (random init)'}")
    print(f" YOLO        : {'LOADED' if (_yolo and _yolo.status.get('loaded')) else 'MISSING'}")
    print(f" Calibration : {'LOADED' if _saved_homography else 'NONE'}")
    _bw, _bh = _get_board_bounds()
    print(f" Board size  : {_bw:g} x {_bh:g} mm ({'from calibration' if _saved_board_size else 'default'})")
    print(f" Classes     : {len(CLASS_NAMES)}")
    print(f" Device      : {_device}")
    if _align_cnn is not None:
        print(f" JEPA heads  : TRAINED (synthetic val angle_MAE={_jepa_heads_meta['align_metrics']['val_angle_mae_deg']}deg,"
              f" validate_acc={_jepa_heads_meta['validate_metrics']['val_accuracy']}) - not yet validated on real captures")
    else:
        print(" JEPA heads  : none trained - /nn/align /nn/validate use classical-CV fallback (run: python jepa_heads.py)")
    print(f" URL         : http://127.0.0.1:5000")
    print(" New routes  : /nn/align /nn/validate /schematic/generate /schematic/drc")
    print("               /layout/place /layout/route")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)