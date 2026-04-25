"""
pcb_jepa_nn.py — PCBWorkspace SERC
Multi-task CNN using pretrained ResNet18 backbone.
Fine-tuned for PCB component detection and defect detection.
"""

from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@dataclass
class JEPAConfig:
    dropout: float = 0.3
    component_classes: List[str] = field(default_factory=lambda: [
        "Resistor", "Capacitor", "Diode", "LED", "Transistor",
        "Channel Port", "IC", "Crystal", "Inductor", "Fuse",
        "Button", "Connector", "SOT-23", "QFP", "BGA",
        "0402", "0603", "0805", "1206", "SOD-123",
    ])
    defect_classes: List[str] = field(default_factory=lambda: [
        "none", "missing", "misaligned", "wrong_component",
        "solder_bridge", "tombstone",
    ])

    @property
    def num_components(self): return len(self.component_classes)
    @property
    def num_defects(self): return len(self.defect_classes)


class PCBVisionSystem(nn.Module):
    """
    Pretrained ResNet18 backbone + three task heads:
      - ComponentHead: classify component type + bbox
      - DefectHead:    classify defect type
      - AlignmentHead: predict dx, dy, dtheta
    """
    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg

        # Load pretrained ResNet18 — real-world visual features out of the box
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = resnet.fc.in_features  # 512

        # Remove the final classification layer — use as feature extractor
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feat_dim = feat_dim

        # Component head
        self.comp_shared = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout)
        )
        self.comp_cls  = nn.Linear(256, cfg.num_components)
        self.comp_bbox = nn.Sequential(nn.Linear(256, 4), nn.Sigmoid())

        # Defect head
        self.defect_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout), nn.Linear(128, cfg.num_defects)
        )

        # Alignment head
        self.align_head = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout), nn.Linear(128, 64),
            nn.ReLU(inplace=True), nn.Linear(64, 3)
        )

    def forward(self, x):
        feat = self.backbone(x).flatten(1)  # (B, 512)
        h = self.comp_shared(feat)
        return {
            "comp_logits":   self.comp_cls(h),
            "bbox":          self.comp_bbox(h),
            "defect_logits": self.defect_head(feat),
            "alignment":     self.align_head(feat),
        }

    @torch.no_grad()
    def infer_detect(self, x):
        out        = self.forward(x)
        comp_probs = F.softmax(out["comp_logits"], dim=-1)[0]
        comp_idx   = int(comp_probs.argmax())
        def_probs  = F.softmax(out["defect_logits"], dim=-1)[0]
        def_idx    = int(def_probs.argmax())
        return {
            "class_name":  self.cfg.component_classes[comp_idx],
            "class_idx":   comp_idx,
            "confidence":  round(float(comp_probs[comp_idx]), 4),
            "bbox":        [round(float(v), 4) for v in out["bbox"][0]],
            "defect":      self.cfg.defect_classes[def_idx],
            "defect_conf": round(float(def_probs[def_idx]), 4),
        }

    @torch.no_grad()
    def infer_align(self, x):
        out = self.forward(x)
        a   = out["alignment"][0]
        return {
            "delta_x_mm":      round(float(a[0]) * 0.5, 4),
            "delta_y_mm":      round(float(a[1]) * 0.5, 4),
            "delta_theta_deg": round(float(a[2]) * 5.0, 3),
            "confidence":      round(float(F.softmax(out["defect_logits"], dim=-1)[0][0]), 4),
        }

    @torch.no_grad()
    def infer_validate(self, x):
        out       = self.forward(x)
        def_probs = F.softmax(out["defect_logits"], dim=-1)[0]
        pass_prob = float(def_probs[0])
        return {
            "decision":  "PASS" if pass_prob >= 0.5 else "FAIL",
            "pass_prob": round(pass_prob, 4),
            "fail_prob": round(1.0 - pass_prob, 4),
            "defect":    self.cfg.defect_classes[int(def_probs.argmax())],
        }


def multitask_loss(outputs, comp_labels, bbox_targets, defect_labels,
                   w_comp=1.0, w_bbox=1.0, w_defect=1.5):
    l_comp   = F.cross_entropy(outputs["comp_logits"],   comp_labels)
    l_bbox   = F.smooth_l1_loss(outputs["bbox"],         bbox_targets)
    l_defect = F.cross_entropy(outputs["defect_logits"], defect_labels)
    total    = w_comp * l_comp + w_bbox * l_bbox + w_defect * l_defect
    return total, {
        "l_comp":   l_comp.item(),
        "l_bbox":   l_bbox.item(),
        "l_defect": l_defect.item(),
    }


if __name__ == "__main__":
    cfg   = JEPAConfig()
    model = PCBVisionSystem(cfg)
    x     = torch.randn(2, 3, 224, 224)
    out   = model(x)
    print("comp_logits:", out["comp_logits"].shape)
    print("bbox       :", out["bbox"].shape)
    print("defect     :", out["defect_logits"].shape)
    print("alignment  :", out["alignment"].shape)
    print("params     :", f"{sum(p.numel() for p in model.parameters()):,}")
    print("detect     :", model.infer_detect(x[:1]))
