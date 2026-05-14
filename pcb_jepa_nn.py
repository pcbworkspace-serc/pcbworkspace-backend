"""
pcb_jepa_nn.py — PCBWorkspace SERC backend

MobileNetV3-Small multi-label PCB component classifier + multi-scale
sliding-window detector with per-class NMS.

Matches the model trained in PCB_Component_Detection_MobileNetV3.ipynb
(ramalliss/pcb-component-detection). Paper-reported val mAP = 0.636.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ── Class taxonomy ─────────────────────────────────────────────────────────────
# CLASS_NAMES order MUST match the training notebook — the checkpoint's output
# dimension is indexed by this order.
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

# ── Preprocessing (must match training) ────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
WIN_SIZE = 64  # model trained on 64×64 patches

# ── Sliding-window detection params (from notebook) ────────────────────────────
WORK_SIZES = [256, 384, 576]
STRIDE = 32
SCORE_MIN = 0.25
NMS_IOU = 0.30
TOP_K_PER_WIN = 3
MAX_BOXES = 40


# ── Config (name kept for backwards compat with old flask_server imports) ──────
@dataclass
class JEPAConfig:
    dropout: float = 0.3
    component_classes: List[str] = field(default_factory=lambda: list(CLASS_NAMES))

    @property
    def num_components(self) -> int:
        return len(self.component_classes)


def _build_mobilenet_v3(num_classes: int, dropout: float = 0.3) -> nn.Module:
    """MobileNetV3-Small with a fresh num_classes head (matches training)."""
    # weights=None — real weights come from the checkpoint
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    if isinstance(model.classifier[2], nn.Dropout):
        model.classifier[2].p = dropout
    return model


class PCBVisionSystem(nn.Module):
    """
    Multi-label PCB classifier + multi-scale sliding-window detector.
    Wraps a MobileNetV3-Small backbone so the existing flask_server import
    `from pcb_jepa_nn import PCBVisionSystem` continues to work.
    """

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

    # ── Inference: whole-image multi-label classification ──────────────────────
    @torch.no_grad()
    def infer_multilabel(self, pil_image: Image.Image) -> Dict:
        """Returns sigmoid-scored predictions sorted descending."""
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
        return {
            "predictions": predictions,
            "num_classes": len(self.class_names),
        }

    # ── Inference: multi-scale sliding window with per-class NMS ───────────────
    @torch.no_grad()
    def detect_boxes(self, pil_image: Image.Image) -> Dict:
        self.eval()
        device = next(self.parameters()).device
        orig_w, orig_h = pil_image.size
        all_cands, total_windows = [], 0
        for ws in WORK_SIZES:
            cands, nwin = self._detect_at_scale(pil_image, ws, orig_w, orig_h, device)
            all_cands.extend(cands)
            total_windows += nwin
        kept = self._nms(all_cands)[:MAX_BOXES]
        for b in kept:
            b["class_full"] = FULL_NAMES.get(b["class"], b["class"])
            x1, y1, x2, y2 = b["box"]
            b["box_norm"] = [
                round(x1 / max(1, orig_w), 4), round(y1 / max(1, orig_h), 4),
                round(x2 / max(1, orig_w), 4), round(y2 / max(1, orig_h), 4),
            ]
            b["box"] = [round(v, 2) for v in b["box"]]
            b["score"] = round(float(b["score"]), 4)
        return {
            "boxes": kept,
            "image_size": [orig_w, orig_h],
            "n_windows_evaluated": total_windows,
        }

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
                    "box": [x * inv, y * inv,
                            (x + WIN_SIZE) * inv, (y + WIN_SIZE) * inv],
                    "class": self.class_names[int(ci)],
                    "score": s,
                    "scale": work_size,
                })
        return cands, len(positions)

    @staticmethod
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

    def _nms(self, boxes):
        by_class = {}
        for b in boxes:
            by_class.setdefault(b["class"], []).append(b)
        kept = []
        for _, items in by_class.items():
            items.sort(key=lambda x: -x["score"])
            survivors = []
            for cand in items:
                if all(self._iou(cand["box"], s["box"]) <= NMS_IOU for s in survivors):
                    survivors.append(cand)
            kept.extend(survivors)
        kept.sort(key=lambda x: -x["score"])
        return kept


def load_model(checkpoint_path: str = "best.pt",
               device: str = "cpu") -> Tuple[PCBVisionSystem, bool]:
    """Build the model and load weights from `checkpoint_path`."""
    cfg = JEPAConfig()
    model = PCBVisionSystem(cfg)
    ckpt_file = Path(checkpoint_path)
    loaded = False
    if ckpt_file.exists():
        try:
            state = torch.load(ckpt_file, map_location=device)
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            # Notebook saves raw MobileNetV3 state dict
            try:
                model.backbone.load_state_dict(state)
                loaded = True
            except RuntimeError:
                # Fall back to the wrapped state dict format
                model.load_state_dict(state)
                loaded = True
        except Exception as e:
            print(f"  WARN: failed to load checkpoint {ckpt_file}: {e}")
    model.to(device).eval()
    return model, loaded


# Backwards-compat stub — old flask_server.py imported this
def multitask_loss(*args, **kwargs):
    raise NotImplementedError(
        "Training disabled in this build. "
        "Retrain via the Colab notebook in ramalliss/pcb-component-detection."
    )
