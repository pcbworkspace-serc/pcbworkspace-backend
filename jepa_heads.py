"""
jepa_heads.py — synthetic-data-trained AlignmentCorrector / PlacementValidator
heads for /nn/align and /nn/validate, sitting on top of the already-deployed
MobileNetV3 backbone (best.pt) as a frozen feature extractor.

Why this exists: vision_classical.py is the classical-CV fallback (contour +
minAreaRect). This module is the "drop-in replacement" it was designed for —
same response shape, real learned weights instead of geometry heuristics.
It does NOT reuse jepa_checkpoint.pt: that file was audited and found to be
an orphaned artifact from an earlier, incompatible architecture (ResNet18
backbone, 20-class taxonomy vs. the deployed 25-class MobileNetV3 — see repo
history) whose align_head output barely changes between a zero vector and
random input (collapsed / never meaningfully trained). Safer to train fresh
heads against the model that's actually running in production.

Training data: there is no labeled real alignment/defect dataset yet, so
these heads are trained on domain-randomized synthetic component crops with
programmatically known ground truth (rotation, offset, pass/fail failure
mode). This is standard practice for this exact problem — see the
"Prior Availability in Industrial Visual Sim-to-Real" survey and
"Hybrid Synthetic Data Generation with Domain Randomization ... Under
Extreme Class Imbalance" (2025/2026) for the general approach, and the
"Vision-Based 6D Pose Analytics" pick-and-place paper (2025) for the
sub-mm / sub-degree accuracy target this kind of pipeline is aiming at.
Two technique choices came directly from that research:
  - angle is regressed as (sin θ, cos θ) and recovered via atan2, not as a
    raw degree value — avoids the +/-180 deg wraparound discontinuity that
    plain angle regression fights against (see "Image Rotation Angle
    Estimation: Comparing Circular-Aware Methods", 2026)
  - the backbone is frozen and only small MLP heads are trained on top of
    it — the "frozen evaluation" pattern JEPA-family papers converge on,
    rather than fine-tuning the whole backbone on a small synthetic set

IMPORTANT — read before trusting this in production: synthetic-only
training means these numbers describe performance on synthetic renders,
not on your real camera feed. Domain randomization (varied backgrounds,
lighting, blur, occlusion) is there specifically to help it transfer, but
"trained on synthetic data" and "validated on real hardware" are different
claims — this module only supports the first one. Re-validate against real
top/bottom camera captures before leaning on it for production placement
decisions; keep vision_classical.py as the fallback until you do.
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

WIN_SIZE = 64  # matches pcb_jepa_nn.py's WIN_SIZE / _win_transform crop size
FEATURE_DIM = 1024  # mobilenet_v3_small classifier[-1].in_features
MAX_OFFSET_PX = 20.0  # +/- offset range synthesized (relative to a 64px crop)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────
# Feature extractor — wraps the already-loaded, already-trained classifier
# backbone. Frozen: we only ever train the small heads on top of it.
# ─────────────────────────────────────────────────────────────────────────

class FrozenFeatureExtractor:
    def __init__(self, backbone_model):
        """backbone_model: a torchvision mobilenet_v3_small with a replaced
        final classifier layer (i.e. PCBVisionSystem.backbone from
        pcb_jepa_nn.py). We run everything except the final Linear."""
        self.net = backbone_model
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def __call__(self, x):
        feats = self.net.features(x)
        feats = self.net.avgpool(feats)
        feats = torch.flatten(feats, 1)
        # classifier is Sequential(Linear, Hardswish, Dropout, Linear[replaced]);
        # index everything up to (not including) the final replaced Linear
        for layer in list(self.net.classifier.children())[:-1]:
            feats = layer(feats)
        return feats  # (B, FEATURE_DIM)


# ─────────────────────────────────────────────────────────────────────────
# Heads
# ─────────────────────────────────────────────────────────────────────────
#
# AlignHead does NOT sit on the frozen classifier backbone, unlike
# ValidateHead below. First version did (see git history) and the angle
# regression never learned past ~88deg MAE — indistinguishable from a
# constant predictor on a uniform [-180,180] target. That's not a training
# bug, it's an architecture mismatch: the classifier backbone was trained
# for component *type*, and type is rotation-invariant (a resistor is a
# resistor at any angle), so its own training augmentation actively
# teaches it to discard orientation as a nuisance variable. You cannot
# regress an angle from features whose whole point was to not encode it.
# AlignCNN is a small dedicated backbone trained end-to-end on the
# alignment task instead, so it's free to keep the orientation signal the
# classifier backbone throws away.

class AlignCNN(nn.Module):
    """Small conv trunk trained end-to-end for rotation/offset regression —
    deliberately separate from the (rotation-invariant) classifier backbone.

    Returns the raw (B, C, 4, 4) spatial feature map, NOT a pooled vector.
    First version pooled here (nn.AdaptiveAvgPool2d(1)) and rotation trained
    fine, but dx/dy offset flatlined exactly at the "always predict zero"
    baseline (MAE == E[|Uniform(-20,20)|] == 10px, unmoving for 30 epochs)
    — global average pooling is *defined* to be translation-invariant, so
    it throws away the one thing offset regression needs (WHERE something
    is), the same way the frozen classifier backbone threw away rotation.
    Same bug, same fix: don't let anything before the offset head destroy
    the signal that head needs to read."""
    def __init__(self, out_channels=128):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.trunk = nn.Sequential(
            block(3, 16),           # 64 -> 32
            block(16, 32),          # 32 -> 16
            block(32, 64),          # 16 -> 8
            block(64, out_channels),  # 8 -> 4
        )

    def forward(self, x):
        return self.trunk(x)  # (B, out_channels, 4, 4)


class AlignHead(nn.Module):
    """Outputs (sin_theta, cos_theta, dx_norm, dy_norm). dx/dy are in
    [-1, 1], scaled to +/- MAX_OFFSET_PX pixels by the caller.

    Two separate branches reading the same spatial feature map differently:
    rotation from a pooled (position-blind, orientation-sensitive) summary,
    offset from the flattened spatial map (position-preserving) — pooling
    the offset branch reproduces the flatlined-at-zero bug above."""
    def __init__(self, in_channels=128, spatial=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.rot_net = nn.Sequential(
            nn.Linear(in_channels, 64), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 2),
        )
        flat_dim = in_channels * spatial * spatial
        self.offset_net = nn.Sequential(
            nn.Linear(flat_dim, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 32), nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(self, feat_map):
        pooled = self.pool(feat_map).flatten(1)
        sin_cos = self.rot_net(pooled)
        sin_cos = sin_cos / (sin_cos.norm(dim=-1, keepdim=True) + 1e-6)  # project onto unit circle

        flat = feat_map.flatten(1)
        offset = torch.tanh(self.offset_net(flat))
        return torch.cat([sin_cos, offset], dim=-1)


class ValidateHead(nn.Module):
    """Binary classifier: logits for [fail, pass]."""
    def __init__(self, in_dim=FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 2),
        )

    def forward(self, feats):
        return self.net(feats)


# ─────────────────────────────────────────────────────────────────────────
# Synthetic data generation — domain-randomized rectangular SMD-like blobs
# ─────────────────────────────────────────────────────────────────────────

def _random_bg(size):
    """PCB-ish background: a solid base color (green/brown/black soldermask
    family) plus per-pixel noise and an optional soft vignette/gradient, so
    the head can't just threshold on a fixed background value."""
    base_choices = [(30, 90, 45), (20, 20, 20), (70, 45, 25), (15, 60, 90)]
    base = random.choice(base_choices)
    jitter = tuple(max(0, min(255, c + random.randint(-15, 15))) for c in base)
    arr = np.zeros((size, size, 3), dtype=np.float32)
    arr[:, :] = jitter
    noise = np.random.normal(0, random.uniform(3, 12), (size, size, 3))
    arr += noise
    return np.clip(arr, 0, 255).astype(np.uint8)


def _draw_component(img, center, wh, angle_deg, color, polarity_marker=True):
    """A plain rectangle looks identical at angle and angle+180 — real SMD
    parts don't have that ambiguity because they carry a polarity/pin-1
    marker (diode cathode band, IC pin-1 dot, LED flat edge, ...), which is
    exactly what makes rotation recoverable at all. Draw one so the
    synthetic label is actually recoverable from the pixels, matching the
    real task instead of a strictly-easier one."""
    draw = ImageDraw.Draw(img)
    cx, cy = center
    w, h = wh
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    def rot(px, py):
        return (cx + px * cos_a - py * sin_a, cy + px * sin_a + py * cos_a)
    poly = [rot(px, py) for px, py in pts]
    draw.polygon(poly, fill=color)

    if polarity_marker:
        marker_color = tuple(255 - c for c in color)  # high-contrast vs. body color
        mx, my = rot(-w / 2 * 0.72, 0)  # band near the "pin 1" short edge
        mr = max(1.2, min(w, h) * 0.12)
        draw.ellipse([mx - mr, my - mr, mx + mr, my + mr], fill=marker_color)
    return img


def _post_process(img, blur_p=0.4, occlude_p=0.15):
    if random.random() < blur_p:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
    if random.random() < occlude_p:
        draw = ImageDraw.Draw(img)
        ox, oy = random.uniform(0, WIN_SIZE), random.uniform(0, WIN_SIZE)
        ow, oh = random.uniform(4, 14), random.uniform(4, 14)
        occ_color = tuple(random.randint(0, 255) for _ in range(3))
        draw.rectangle([ox, oy, ox + ow, oy + oh], fill=occ_color)
    arr = np.array(img).astype(np.float32)
    brightness = random.uniform(0.75, 1.25)
    arr = np.clip(arr * brightness, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def gen_alignment_sample():
    """Returns (PIL image, theta_deg, dx_px, dy_px). theta in (-180, 180],
    dx/dy relative to frame center, magnitude up to MAX_OFFSET_PX."""
    bg = _random_bg(WIN_SIZE)
    img = Image.fromarray(bg)
    theta = random.uniform(-180, 180)
    dx = random.uniform(-MAX_OFFSET_PX, MAX_OFFSET_PX)
    dy = random.uniform(-MAX_OFFSET_PX, MAX_OFFSET_PX)
    w = random.uniform(20, 40)
    h = random.uniform(10, w * 0.75)
    comp_color = tuple(random.randint(150, 255) for _ in range(3)) if random.random() < 0.5 \
        else tuple(random.randint(0, 60) for _ in range(3))
    cx, cy = WIN_SIZE / 2 + dx, WIN_SIZE / 2 + dy
    img = _draw_component(img, (cx, cy), (w, h), theta, comp_color)
    img = _post_process(img)
    return img, theta, dx, dy


def gen_validation_sample():
    """Returns (PIL image, label) where label in {"pass","fail"} and a
    sub-reason for fail cases, covering the failure modes PlacementValidator
    is meant to catch: missing, tombstoned, wrong-size, badly misaligned,
    bridged/double part."""
    bg = _random_bg(WIN_SIZE)
    img = Image.fromarray(bg)
    comp_color = tuple(random.randint(150, 255) for _ in range(3)) if random.random() < 0.5 \
        else tuple(random.randint(0, 60) for _ in range(3))

    outcome = random.choices(
        ["pass", "missing", "tombstone", "wrong_size", "misaligned", "bridged"],
        weights=[40, 12, 12, 12, 12, 12],
    )[0]

    if outcome == "missing":
        pass  # blank background only
    elif outcome == "tombstone":
        w, h = random.uniform(6, 10), random.uniform(6, 10)  # part standing on end -> tiny footprint
        img = _draw_component(img, (WIN_SIZE / 2, WIN_SIZE / 2), (w, h), random.uniform(-180, 180), comp_color)
    elif outcome == "wrong_size":
        # much bigger or much smaller than a correctly placed part
        scale = random.choice([random.uniform(2.2, 3.0), random.uniform(0.15, 0.3)])
        w, h = 28 * scale, 16 * scale
        img = _draw_component(img, (WIN_SIZE / 2, WIN_SIZE / 2), (w, h), random.uniform(-180, 180), comp_color)
    elif outcome == "misaligned":
        w, h = random.uniform(20, 32), random.uniform(10, 20)
        dx = random.uniform(18, 30) * random.choice([-1, 1])
        dy = random.uniform(18, 30) * random.choice([-1, 1])
        img = _draw_component(img, (WIN_SIZE / 2 + dx, WIN_SIZE / 2 + dy), (w, h), random.uniform(-180, 180), comp_color)
    elif outcome == "bridged":
        w, h = random.uniform(18, 26), random.uniform(10, 16)
        img = _draw_component(img, (WIN_SIZE / 2 - w * 0.3, WIN_SIZE / 2), (w, h), random.uniform(-15, 15), comp_color)
        img = _draw_component(img, (WIN_SIZE / 2 + w * 0.3, WIN_SIZE / 2), (w, h), random.uniform(-15, 15), comp_color)
    else:  # pass
        w, h = random.uniform(20, 32), random.uniform(10, 20)
        dx = random.uniform(-4, 4)
        dy = random.uniform(-4, 4)
        img = _draw_component(img, (WIN_SIZE / 2 + dx, WIN_SIZE / 2 + dy), (w, h), random.uniform(-180, 180), comp_color)

    img = _post_process(img)
    label = "pass" if outcome == "pass" else "fail"
    return img, label, outcome


def _to_tensor(pil_img):
    arr = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


# ─────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────

def _angle_error_deg(pred_sin_cos, true_theta_deg):
    pred_theta = torch.rad2deg(torch.atan2(pred_sin_cos[:, 0], pred_sin_cos[:, 1]))
    diff = (pred_theta - true_theta_deg + 180) % 360 - 180
    return diff.abs()


def train_align_head(n_train=5000, n_val=1000, epochs=35,
                      batch_size=128, lr=1.5e-3, seed=0, verbose=True):
    """Trains AlignCNN + AlignHead end-to-end (not on frozen classifier
    features — see the comment above AlignCNN for why)."""
    random.seed(seed)
    torch.manual_seed(seed)

    def make_batch(n):
        imgs, thetas, dxs, dys = [], [], [], []
        for _ in range(n):
            img, theta, dx, dy = gen_alignment_sample()
            imgs.append(_to_tensor(img))
            thetas.append(theta)
            dxs.append(dx / MAX_OFFSET_PX)
            dys.append(dy / MAX_OFFSET_PX)
        return (torch.stack(imgs), torch.tensor(thetas, dtype=torch.float32),
                torch.tensor(dxs, dtype=torch.float32), torch.tensor(dys, dtype=torch.float32))

    train_x, train_theta, train_dx, train_dy = make_batch(n_train)
    val_x, val_theta, val_dx, val_dy = make_batch(n_val)

    cnn = AlignCNN()
    head = AlignHead()
    params = list(cnn.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = train_x.shape[0]

    for epoch in range(epochs):
        cnn.train(); head.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            imgs = train_x[idx]
            theta_t = train_theta[idx]
            target_sin = torch.sin(torch.deg2rad(theta_t))
            target_cos = torch.cos(torch.deg2rad(theta_t))
            target_dx = train_dx[idx]
            target_dy = train_dy[idx]

            feats = cnn(imgs)
            out = head(feats)
            loss_ang = F.mse_loss(out[:, 0], target_sin) + F.mse_loss(out[:, 1], target_cos)
            loss_off = F.smooth_l1_loss(out[:, 2], target_dx) + F.smooth_l1_loss(out[:, 3], target_dy)
            loss = loss_ang + loss_off

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        sched.step()

        cnn.eval(); head.eval()
        with torch.no_grad():
            val_out = head(cnn(val_x))
            val_ang_err = _angle_error_deg(val_out[:, :2], val_theta).mean().item()
            val_dx_err = (val_out[:, 2] - val_dx).abs().mean().item() * MAX_OFFSET_PX
            val_dy_err = (val_out[:, 3] - val_dy).abs().mean().item() * MAX_OFFSET_PX
        if verbose:
            print(f"  [align] epoch {epoch+1:>2}/{epochs}  train_loss={total_loss/n:.4f}"
                  f"  val_angle_MAE={val_ang_err:.2f}deg  val_dx_MAE={val_dx_err:.2f}px  val_dy_MAE={val_dy_err:.2f}px")

    metrics = {"val_angle_mae_deg": round(val_ang_err, 3),
               "val_dx_mae_px": round(val_dx_err, 3), "val_dy_mae_px": round(val_dy_err, 3),
               "n_train": n_train, "n_val": n_val, "epochs": epochs}
    return cnn, head, metrics


def train_validate_head(feature_extractor, n_train=4000, n_val=800, epochs=12,
                         batch_size=64, lr=2e-3, seed=0, verbose=True):
    random.seed(seed + 1)
    torch.manual_seed(seed + 1)
    label_to_idx = {"fail": 0, "pass": 1}

    def make_batch(n):
        imgs, labels, reasons = [], [], []
        for _ in range(n):
            img, label, reason = gen_validation_sample()
            imgs.append(_to_tensor(img))
            labels.append(label_to_idx[label])
            reasons.append(reason)
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long), reasons

    train_x, train_y, _ = make_batch(n_train)
    val_x, val_y, val_reasons = make_batch(n_val)

    with torch.no_grad():
        train_feats = torch.cat([feature_extractor(train_x[i:i + 256]) for i in range(0, len(train_x), 256)])
        val_feats = torch.cat([feature_extractor(val_x[i:i + 256]) for i in range(0, len(val_x), 256)])

    head = ValidateHead()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    n = train_feats.shape[0]

    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = head(train_feats[idx])
            loss = F.cross_entropy(logits, train_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats)
            val_pred = val_logits.argmax(dim=-1)
            val_acc = (val_pred == val_y).float().mean().item()
        if verbose:
            print(f"  [validate] epoch {epoch+1:>2}/{epochs}  train_loss={total_loss/n:.4f}  val_acc={val_acc:.4f}")

    # per-failure-mode recall, so "accuracy" isn't hiding a blind spot on
    # one specific defect type
    with torch.no_grad():
        val_pred = head(val_feats).argmax(dim=-1)
    per_reason = {}
    for reason in set(val_reasons):
        mask = [r == reason for r in val_reasons]
        idxs = [i for i, m in enumerate(mask) if m]
        if not idxs:
            continue
        correct = sum(1 for i in idxs if val_pred[i].item() == val_y[i].item())
        per_reason[reason] = round(correct / len(idxs), 3)

    metrics = {"val_accuracy": round(val_acc, 4), "per_failure_mode_recall": per_reason,
               "n_train": n_train, "n_val": n_val, "epochs": epochs}
    return head, metrics


# ─────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "jepa_heads_synthetic.pt"


def save_checkpoint(align_cnn, align_head, validate_head, align_metrics, validate_metrics, path=CHECKPOINT_PATH):
    torch.save({
        "align_cnn_state": align_cnn.state_dict(),
        "align_head_state": align_head.state_dict(),
        "validate_head_state": validate_head.state_dict(),
        "align_metrics": align_metrics,
        "validate_metrics": validate_metrics,
        "feature_dim": FEATURE_DIM,
        "win_size": WIN_SIZE,
        "max_offset_px": MAX_OFFSET_PX,
        "training_data": "synthetic domain-randomized renders only, see jepa_heads.py docstring",
    }, path)


def load_heads(path=CHECKPOINT_PATH):
    """Returns (align_cnn, align_head, validate_head, metadata) or
    (None, None, None, None) if the checkpoint doesn't exist / fails to load."""
    from pathlib import Path
    if not Path(path).exists():
        return None, None, None, None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        align_cnn = AlignCNN()
        align_cnn.load_state_dict(ckpt["align_cnn_state"])
        align_cnn.eval()
        align_head = AlignHead()
        align_head.load_state_dict(ckpt["align_head_state"])
        align_head.eval()
        validate_head = ValidateHead()
        validate_head.load_state_dict(ckpt["validate_head_state"])
        validate_head.eval()
        meta = {k: v for k, v in ckpt.items()
                if k not in ("align_cnn_state", "align_head_state", "validate_head_state")}
        return align_cnn, align_head, validate_head, meta
    except Exception as e:
        print(f"  WARN: failed to load {path}: {e}")
        return None, None, None, None


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from pcb_jepa_nn import load_model

    print("=" * 60)
    print(" Training AlignCNN+AlignHead / ValidateHead on synthetic data")
    print("=" * 60)
    model, loaded = load_model("best.pt", device="cpu")
    print(f" backbone loaded from best.pt: {loaded}")
    fx = FrozenFeatureExtractor(model.backbone)

    align_cnn, align_head, align_metrics = train_align_head()
    validate_head, validate_metrics = train_validate_head(fx)

    save_checkpoint(align_cnn, align_head, validate_head, align_metrics, validate_metrics)
    print("-" * 60)
    print(" Saved:", CHECKPOINT_PATH)
    print(" align_metrics   :", align_metrics)
    print(" validate_metrics:", validate_metrics)
    print("=" * 60)
