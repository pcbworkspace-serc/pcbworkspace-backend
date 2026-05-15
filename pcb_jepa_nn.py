"""
pcb_jepa_nn.py — PCBWorkspace SERC backend

Two detection paths:
  1. Sliding window — fixed-grid 64×64 windows at multi-scale, MobileNetV3 classifies each
  2. YOLO hybrid    — YOLOv8n proposes boxes, MobileNetV3 classifies each crop

Frontend lets the user toggle between them.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ── Class taxonomy ─────────────────────────────────────────────────────────────
# Order MUST match the training notebook — checkpoint output dim indexed by this.
CLASS_NAMES: List[str] = [
    "RN", "RA", "U", "FB", "T", "D", "SW", "F", "BTN", "CRA",
    "Q", "QA", "IC", "M", "L", "V", "CR", "S", "P", "TP",
    "LED", "R", "J", "JP", "C",
]

FULL_NAMES: Dict[str, str] = {
    "R": "Resistor", "RN": "Resistor Network", "RA": "Resistor Array",
    "C": "Capacitor", "L": "Inductor", "D": "Diode", "LED": "LED",
    "Q": "Transistor", "QA": "Transistor Array",
    "U": "Integrated Circuit", "IC": "Integrated Circuit",
    "T": "Transformer", "F": "Fuse", "FB": "Ferrite Bead",
    "SW": "Switch", "BTN": "Button", "CR": "Crystal", "CRA": "Crystal Array",
    "J": "Connector", "JP": "Jumper", "M": "Module", "P": "Plug",
    "S": "Sensor", "TP": "Test Point", "V": "Voltage Regulator",
}

# ── Preprocessing (matches training) ───────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
WIN_SIZE = 64

# ── Sliding-window params (from notebook) ──────────────────────────────────────
WORK_SIZES = [256, 384, 576]
STRIDE = 32
SCORE_MIN = 0.25
NMS_IOU = 0.30
TOP_K_PER_WIN = 3
MAX_BOXES = 40

# ── YOLO params ────────────────────────────────────────────────────────────────
YOLO_WEIGHTS = "yolov8_pcb.pt"
YOLO_CONF = 0.25
YOLO_IOU = 0.45
YOLO_MAX_DET = 60


@dataclass
class JEPAConfig:
    dropout: float = 0.3
    component_classes: List[str] = field(default_factory=lambda: list(CLASS_NAMES))

    @property
    def num_components(self) -> int:
        return len(self.component_classes)


def _build_mobilenet_v3(num_classes: int, dropout: float = 0.3) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    if isinstance(model.classifier[2], nn.Dropout):
        model.classifier[2].p = dropout
    return model


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


class PCBVisionSystem(nn.Module):
    """MobileNetV3-Small + sliding-window detection (Method A)."""

    def __init__(self, cfg: JEPAConfig = None):
        super().__init__()
        self.cfg = cfg or JEPAConfig()
        self.backbone = _build_mobilenet_v3(self.cfg.num_components, self.cfg.dropout)
        self.class_names = list(CLASS_NAMES)
        self._win_transform = transforms.Compose([
            transforms.Resize((WIN_SIZE, WIN_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def forward(self, x):
        return self.backbone(x)

    @torch.no_grad()
    def infer_multilabel(self, pil_image: Image.Image) -> Dict:
        self.eval()
        device = next(self.parameters()).device
        x = self._win_transform(pil_image.convert("RGB")).unsqueeze(0).to(device)
        logits = self.backbone(x)[0]
        probs = torch.sigmoid(logits).cpu().numpy()
        order = np.argsort(probs)[::-1]
        predictions = [
            {
                "class": self.class_names[int(i)],
                "score": round(float(probs[int(i)]), 4),
                "above_threshold": bool(float(probs[int(i)]) >= 0.5),
            }
            for i in order
        ]
        return {"predictions": predictions, "num_classes": len(self.class_names)}

    @torch.no_grad()
    def detect_boxes_sliding(self, pil_image: Image.Image) -> Dict:
        self.eval()
        device = next(self.parameters()).device
        orig_w, orig_h = pil_image.size
        all_cands, total_windows = [], 0
        for ws in WORK_SIZES:
            cands, nwin = self._detect_at_scale(pil_image, ws, orig_w, orig_h, device)
            all_cands.extend(cands)
            total_windows += nwin
        kept = self._nms_per_class(all_cands)[:MAX_BOXES]
        for b in kept:
            self._finalize_box(b, orig_w, orig_h)
        return {
            "boxes": kept,
            "image_size": [orig_w, orig_h],
            "n_windows_evaluated": total_windows,
            "method": "sliding_window",
        }

    def detect_boxes(self, pil_image: Image.Image) -> Dict:
        """Back-compat alias for the original endpoint."""
        return self.detect_boxes_sliding(pil_image)

    def _detect_at_scale(self, img_pil, work_size, orig_w, orig_h, device):
        scale = work_size / max(orig_w, orig_h)
        work_w = max(WIN_SIZE, int(round(orig_w * scale)))
        work_h = max(WIN_SIZE, int(round(orig_h * scale)))
        img_work = img_pil.resize((work_w, work_h), Image.BILINEAR)

        xs = list(range(0, max(1, work_w - WIN_SIZE + 1), STRIDE)) or [0]
        if xs and xs[-1] != work_w - WIN_SIZE:
            xs.append(max(0, work_w - WIN_SIZE))
        ys = list(range(0, max(1, work_h - WIN_SIZE + 1), STRIDE)) or [0]
        if ys and ys[-1] != work_h - WIN_SIZE:
            ys.append(max(0, work_h - WIN_SIZE))

        crops, positions = [], []
        for y in ys:
            for x in xs:
                crops.append(self._win_transform(
                    img_work.crop((x, y, x + WIN_SIZE, y + WIN_SIZE))
                ))
                positions.append((x, y))
        if not crops:
            return [], 0

        batch = torch.stack(crops).to(device)
        probs = torch.sigmoid(self.backbone(batch)).cpu().numpy()
        inv = 1.0 / scale
        cands = []
        for i, (x, y) in enumerate(positions):
            scores = probs[i]
            top = np.argsort(scores)[::-1][:TOP_K_PER_WIN]
            for ci in top:
                s = float(scores[ci])
                if s < SCORE_MIN:
                    continue
                cands.append({
                    "box": [x * inv, y * inv, (x + WIN_SIZE) * inv, (y + WIN_SIZE) * inv],
                    "class": self.class_names[int(ci)],
                    "score": s,
                    "scale": work_size,
                })
        return cands, len(positions)

    @staticmethod
    def _nms_per_class(boxes):
        by_class = {}
        for b in boxes:
            by_class.setdefault(b["class"], []).append(b)
        kept = []
        for items in by_class.values():
            items.sort(key=lambda x: -x["score"])
            survivors = []
            for cand in items:
                if all(_iou(cand["box"], s["box"]) <= NMS_IOU for s in survivors):
                    survivors.append(cand)
            kept.extend(survivors)
        kept.sort(key=lambda x: -x["score"])
        return kept

    @staticmethod
    def _finalize_box(b, orig_w, orig_h):
        b["class_full"] = FULL_NAMES.get(b["class"], b["class"])
        x1, y1, x2, y2 = b["box"]
        b["box_norm"] = [
            round(x1 / max(1, orig_w), 4), round(y1 / max(1, orig_h), 4),
            round(x2 / max(1, orig_w), 4), round(y2 / max(1, orig_h), 4),
        ]
        b["box"] = [round(v, 2) for v in b["box"]]
        b["score"] = round(float(b["score"]), 4)


class YOLOHybridDetector:
    """YOLOv8 box proposer + MobileNetV3 classifier."""

    def __init__(self, classifier: PCBVisionSystem, weights_path: str = YOLO_WEIGHTS):
        self.classifier = classifier
        self.weights_path = Path(weights_path)
        self._yolo = None
        self._import_error: Optional[str] = None

    @property
    def status(self) -> Dict:
        return {
            "weights_exists": self.weights_path.exists(),
            "weights_path": str(self.weights_path),
            "loaded": self._yolo is not None,
            "import_error": self._import_error,
        }

    @property
    def available(self) -> bool:
        if self._yolo is not None:
            return True
        if self._import_error is not None:
            return False
        return self.weights_path.exists()

    def _ensure_loaded(self):
        if self._yolo is not None:
            return self._yolo
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {self.weights_path}. "
                f"Train via PCB_YOLO_Detection.ipynb and place yolov8_pcb.pt in the backend folder."
            )
        try:
            from ultralytics import YOLO
        except ImportError as e:
            self._import_error = str(e)
            raise ImportError(
                "ultralytics not installed. Add 'ultralytics>=8.0' to requirements.txt."
            ) from e
        self._yolo = YOLO(str(self.weights_path))
        return self._yolo

    @torch.no_grad()
    def detect(self, pil_image: Image.Image) -> Dict:
        yolo = self._ensure_loaded()
        orig_w, orig_h = pil_image.size

        results = yolo.predict(
            np.array(pil_image.convert("RGB")),
            conf=YOLO_CONF, iou=YOLO_IOU, max_det=YOLO_MAX_DET,
            verbose=False,
        )[0]
        if len(results.boxes) == 0:
            return {
                "boxes": [], "image_size": [orig_w, orig_h],
                "n_proposals": 0, "method": "yolo_hybrid",
            }

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        yolo_scores = results.boxes.conf.cpu().numpy()

        device = next(self.classifier.parameters()).device
        tf = self.classifier._win_transform
        crops = []
        for (x1, y1, x2, y2) in boxes_xyxy:
            crop = pil_image.crop((float(x1), float(y1), float(x2), float(y2)))
            crops.append(tf(crop))
        batch = torch.stack(crops).to(device)
        probs = torch.sigmoid(self.classifier.backbone(batch)).cpu().numpy()

        out_boxes = []
        for (x1, y1, x2, y2), yscore, p in zip(boxes_xyxy, yolo_scores, probs):
            top_idx = int(np.argmax(p))
            cls = self.classifier.class_names[top_idx]
            cls_score = float(p[top_idx])
            # Combined score = geometric mean of localization + classification
            combined = float(np.sqrt(float(yscore) * cls_score))
            x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
            out_boxes.append({
                "box": [round(x1f, 2), round(y1f, 2), round(x2f, 2), round(y2f, 2)],
                "box_norm": [
                    round(x1f / max(1, orig_w), 4), round(y1f / max(1, orig_h), 4),
                    round(x2f / max(1, orig_w), 4), round(y2f / max(1, orig_h), 4),
                ],
                "class": cls,
                "class_full": FULL_NAMES.get(cls, cls),
                "score": round(combined, 4),
                "yolo_score": round(float(yscore), 4),
                "classifier_score": round(cls_score, 4),
            })
        out_boxes.sort(key=lambda b: -b["score"])
        return {
            "boxes": out_boxes,
            "image_size": [orig_w, orig_h],
            "n_proposals": len(out_boxes),
            "method": "yolo_hybrid",
        }


def load_model(checkpoint_path: str = "best.pt",
               device: str = "cpu") -> Tuple[PCBVisionSystem, bool]:
    cfg = JEPAConfig()
    model = PCBVisionSystem(cfg)
    ckpt_file = Path(checkpoint_path)
    loaded = False
    if ckpt_file.exists():
        try:
            state = torch.load(ckpt_file, map_location=device)
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            try:
                model.backbone.load_state_dict(state)
                loaded = True
            except RuntimeError:
                model.load_state_dict(state)
                loaded = True
        except Exception as e:
            print(f"  WARN: failed to load checkpoint {ckpt_file}: {e}")
    model.to(device).eval()
    return model, loaded


def load_yolo_detector(classifier: PCBVisionSystem,
                       weights_path: str = YOLO_WEIGHTS) -> YOLOHybridDetector:
    return YOLOHybridDetector(classifier, weights_path)


def multitask_loss(*args, **kwargs):
    raise NotImplementedError("Training disabled in this build.")
