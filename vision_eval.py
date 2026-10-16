"""
vision_eval.py — held-out evaluation of the vision stack on real SERC frames
that a person has labeled (the captures data_capture.py saves and
/nn/feedback labels).

Until this runs on such frames, SERC has no detection or alignment number of
its own: /nn/status only carries the FPIC paper's figure, and jepa_heads.py
only has synthetic-validation numbers. This harness produces the first real
one, and is built to refuse rather than produce a misleading one.

What it measures, on the held-out split only:
  detection  — per-class and pooled precision/recall of each detector
               (YOLOv8n + MobileNetV3 hybrid; the sliding-window path on
               request) against human-drawn boxes, greedy one-to-one matching
               at IoU >= --iou within each class.
  alignment  — classical-CV (vision_classical.py) vs. the synthetic-trained
               JEPA head (jepa_heads.py) on the same frames: angle error in
               degrees, center-offset error in mm, how often each path
               answered at all, and the paired JEPA-minus-classical gap.

Confidence intervals (95%):
  * Wilson score interval for every proportion. It treats boxes as
    independent, which boxes on one board are not, so it runs narrow.
  * Percentile bootstrap that resamples whole boards/sessions, which keeps
    that correlation. With fewer than 5 held-out boards/sessions it falls
    back to resampling frames and says so — that interval is too narrow.

What makes it refuse (exit 2, nothing reported):
  * any held-out label not made by a person: label_source must be "human"
    with a labeler named (see /nn/feedback). Labels left unspecified,
    marked "model" or "automated", or identical to the model's own
    prediction stored at capture time all count as not human.
  * leakage: a held-out board/session or capture id that a model was
    trained on (--trained-on), or one image filed under two boards/sessions.
  * malformed ground truth or a missing image file.
  * synthetic and real frames mixed in one capture dir. A dir that is all
    synthetic (capture_context.synthetic, as the tests write) is evaluated
    but the report says SYNTHETIC on every top-level field that matters.

The split is by board/session, never by frame: frames that share a
board_id or a session_id are joined into one group (union-find), and a
group is held out when any of its ids hashes into the test fraction. A
group's side therefore depends only on its own ids and the seed, never on
what else is in the dir, and new data can only move a group from the
reserved side to the held-out side, not back. Frames with neither id are
left out and counted. --write-split saves which ids are reserved for
fine-tuning, so a future training run can stay off the held-out boards.

Usage:
    python vision_eval.py --capture-dir data_capture --out vision_eval_report.json
    python vision_eval.py --help
"""

import argparse
import datetime
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import data_capture

HARNESS_VERSION = "1"
Z_95 = 1.959963984540054
EVAL_KINDS = ("detect", "align")
REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class EvalConfig:
    test_fraction: float = 0.3
    split_seed: str = "serc-heldout-v1"
    iou_threshold: float = 0.5
    score_threshold: float = 0.0
    angle_period_deg: float = 360.0
    px_per_mm: float = None
    px_per_mm_source: str = None
    n_boot: int = 2000
    bootstrap_seed: int = 0
    min_groups_for_board_bootstrap: int = 5
    near_duplicate_max_hamming: int = 4


class EvalRefused(Exception):
    """The data can't support an honest number; .problems says why."""

    def __init__(self, reason, problems):
        self.reason = reason
        self.problems = list(problems)
        shown = "; ".join(self.problems[:10])
        more = f" (+{len(self.problems) - 10} more)" if len(self.problems) > 10 else ""
        super().__init__(f"{reason}: {shown}{more}")


# ─────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────

def wilson_interval(successes, n, z=Z_95):
    """Wilson score interval for a binomial proportion; None when n == 0."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # the bound at p = 0 or 1 is exactly 0 or 1; pin it against float round-off
    lo = 0.0 if successes <= 0 else max(0.0, center - half)
    hi = 1.0 if successes >= n else min(1.0, center + half)
    return (lo, hi)


def _weighted_quantile(sorted_values, weights, q):
    """Inverted-CDF quantile (smallest value whose cumulative weight reaches
    q of the total), row-wise over a (B, n) weight matrix whose columns
    follow sorted_values. Same definition for the point estimate (all
    weights 1) and every bootstrap replicate."""
    cum = np.cumsum(weights, axis=1)
    target = q * cum[:, -1:]
    idx = np.argmax(cum >= target - 1e-9, axis=1)
    return sorted_values[idx]


class Resampler:
    """Bootstrap weights over items (frames) that belong to groups
    (boards/sessions). Resamples whole groups when there are enough of
    them, else frames, and records which it did."""

    def __init__(self, item_groups, cfg, rng):
        self.n_items = len(item_groups)
        self.n_groups = len(set(item_groups))
        if self.n_groups >= cfg.min_groups_for_board_bootstrap:
            self.method = "board_bootstrap"
            index = {g: i for i, g in enumerate(sorted(set(item_groups)))}
            unit_of_item = np.array([index[g] for g in item_groups], dtype=int)
            n_units = self.n_groups
        else:
            self.method = "frame_bootstrap"
            unit_of_item = np.arange(self.n_items)
            n_units = self.n_items
        self.weights = None
        if n_units >= 2:
            draws = rng.multinomial(n_units, np.full(n_units, 1.0 / n_units), size=cfg.n_boot)
            self.weights = draws[:, unit_of_item].astype(np.float64)

    @property
    def warning(self):
        if self.method == "frame_bootstrap":
            return (f"only {self.n_groups} held-out board/session group(s): CI resamples frames, "
                    "which ignores within-board correlation, so it is likely too narrow")
        return None

    def ci(self, replicates):
        if replicates is None:
            return None
        reps = np.asarray(replicates, dtype=float)
        reps = reps[np.isfinite(reps)]
        if reps.size == 0:
            return None
        lo, hi = np.percentile(reps, [2.5, 97.5])
        return (float(lo), float(hi))


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _interval(iv, nd=4):
    return None if iv is None else [_r(iv[0], nd), _r(iv[1], nd)]


def summarize_errors(values, item_groups, resampler):
    """mean / median / p95 of non-negative errors, each with a bootstrap
    CI from the given resampler (whose items must be these values)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"n": 0}
    order = np.argsort(v, kind="stable")
    sv = v[order]
    ones = np.ones((1, v.size))
    out = {"n": int(v.size), "n_groups": len(set(item_groups))}
    W = resampler.weights
    stats = {
        "mean": (float(v.mean()), None if W is None else (W @ v) / W.sum(axis=1)),
        "median": (float(_weighted_quantile(sv, ones, 0.5)[0]),
                   None if W is None else _weighted_quantile(sv, W[:, order], 0.5)),
        "p95": (float(_weighted_quantile(sv, ones, 0.95)[0]),
                None if W is None else _weighted_quantile(sv, W[:, order], 0.95)),
    }
    for name, (point, reps) in stats.items():
        out[name] = {"value": _r(point), "ci95": _interval(resampler.ci(reps)), "ci_method": resampler.method}
    return out


def summarize_signed_mean(values, resampler):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None
    W = resampler.weights
    reps = None if W is None else (W @ v) / W.sum(axis=1)
    return {"value": _r(v.mean()), "ci95": _interval(resampler.ci(reps)), "ci_method": resampler.method}


# ─────────────────────────────────────────────────────────────────────────
# Geometry / matching
# ─────────────────────────────────────────────────────────────────────────

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def match_frame(gt_boxes, pred_boxes, iou_threshold):
    """Greedy one-to-one matching within each class: predictions in
    descending score order, each taking the unmatched same-class ground
    truth box with the highest IoU at or above the threshold.
    Returns {class: [tp, fp, fn]}."""
    counts = {}
    classes = {b["class"] for b in gt_boxes} | {p["class"] for p in pred_boxes}
    for cls in classes:
        gts = [b["box"] for b in gt_boxes if b["class"] == cls]
        preds = sorted((p for p in pred_boxes if p["class"] == cls), key=lambda p: -float(p.get("score", 0.0)))
        taken = [False] * len(gts)
        tp = fp = 0
        for p in preds:
            best, best_iou = None, iou_threshold
            for i, g in enumerate(gts):
                if taken[i]:
                    continue
                o = iou(p["box"], g)
                if o >= best_iou:
                    best, best_iou = i, o
            if best is None:
                fp += 1
            else:
                taken[best] = True
                tp += 1
        counts[cls] = [tp, fp, len(gts) - tp]
    return counts


def angle_error_deg(pred, truth, period=360.0):
    """Signed difference wrapped into [-period/2, period/2)."""
    return (pred - truth + period / 2.0) % period - period / 2.0


# ─────────────────────────────────────────────────────────────────────────
# Loading, provenance, ground-truth checks
# ─────────────────────────────────────────────────────────────────────────

def load_records(capture_dir):
    """Every labeled capture of every kind, each tagged with a unique key."""
    records = []
    for kind in data_capture.KINDS:
        for rec in data_capture.load_labeled_dataset(kind, base_dir=str(capture_dir)):
            rec["_key"] = f"{kind}/{rec['capture_id']}"
            records.append(rec)
    return records


def _is_synthetic(rec):
    return bool((rec.get("capture_context") or {}).get("synthetic"))


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_ground_truth(rec):
    """Returns a list of schema problems (empty when the label is usable)."""
    gt = rec.get("ground_truth")
    key = rec["_key"]
    if not isinstance(gt, dict):
        return [f"{key}: ground_truth must be an object"]
    problems = []
    if rec["kind"] == "detect":
        boxes = gt.get("boxes")
        if not isinstance(boxes, list):
            return [f"{key}: detect ground_truth needs a 'boxes' list (empty list = no parts in frame)"]
        for i, b in enumerate(boxes):
            box = b.get("box") if isinstance(b, dict) else None
            cls = b.get("class") if isinstance(b, dict) else None
            if not isinstance(cls, str) or not cls.strip():
                problems.append(f"{key}: boxes[{i}] has no class")
            if not (isinstance(box, list) and len(box) == 4 and all(_finite(v) for v in box)
                    and box[2] > box[0] and box[3] > box[1]):
                problems.append(f"{key}: boxes[{i}].box must be [x1, y1, x2, y2] pixels with x2>x1, y2>y1")
    elif rec["kind"] == "align":
        if not _finite(gt.get("delta_theta_deg")):
            problems.append(f"{key}: align ground_truth needs a numeric delta_theta_deg")
        for unit in ("mm", "px"):
            xs = [gt.get(f"delta_x_{unit}"), gt.get(f"delta_y_{unit}")]
            present = [v is not None for v in xs]
            if any(present) and not all(_finite(v) for v in xs):
                problems.append(f"{key}: delta_x_{unit} and delta_y_{unit} must both be numbers")
        if "px_per_mm" in gt and not (_finite(gt["px_per_mm"]) and gt["px_per_mm"] > 0):
            problems.append(f"{key}: px_per_mm must be a positive number")
    return problems


def _copied_from_model(rec):
    """True when the stored label is the model's own prediction verbatim —
    a pre-filled review form submitted unchanged, which says nothing about
    whether the model was right."""
    gt, pred = rec.get("ground_truth") or {}, rec.get("model_prediction") or {}
    if rec["kind"] == "detect":
        def norm(boxes):
            return sorted((b.get("class"), tuple(round(float(v), 2) for v in b.get("box", []))) for b in boxes)
        gt_boxes, pred_boxes = gt.get("boxes") or [], pred.get("boxes") or []
        return bool(gt_boxes) and norm(gt_boxes) == norm(pred_boxes)
    if rec["kind"] == "align":
        if not pred.get("available", True) or not _finite(pred.get("delta_theta_deg")):
            return False
        fields = [f for f in ("delta_theta_deg", "delta_x_mm", "delta_y_mm", "delta_x_px", "delta_y_px")
                  if _finite(gt.get(f)) and _finite(pred.get(f))]
        return "delta_theta_deg" in fields and all(abs(gt[f] - pred[f]) < 1e-9 for f in fields)
    return False


def check_label_provenance(rec):
    key = rec["_key"]
    source = rec.get("label_source")
    if source != "human":
        shown = "unspecified" if source is None else repr(source)
        return [f"{key}: label_source is {shown}, not 'human'"]
    if not rec.get("labeler"):
        return [f"{key}: human label has no labeler recorded"]
    if _copied_from_model(rec):
        return [f"{key}: ground_truth is identical to the model's own prediction"]
    return []


# ─────────────────────────────────────────────────────────────────────────
# Split by board/session
# ─────────────────────────────────────────────────────────────────────────

def _unit_hash(seed, node):
    digest = hashlib.sha256(f"{seed}|{node}".encode("utf-8")).hexdigest()
    return int(digest[:15], 16) / float(16 ** 15)


def _group_nodes(rec):
    nodes = []
    if rec.get("board_id"):
        nodes.append("board:" + str(rec["board_id"]))
    if rec.get("session_id"):
        nodes.append("session:" + str(rec["session_id"]))
    return nodes


def group_records(records):
    """Union-find over board and session ids. Returns ({key: group_id},
    {group_id: set(nodes)}, [keys with no id]); group_id is the group's
    smallest node name."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    ungrouped = []
    for rec in records:
        nodes = _group_nodes(rec)
        if not nodes:
            ungrouped.append(rec["_key"])
            continue
        for n in nodes:
            parent.setdefault(n, n)
        for n in nodes[1:]:
            union(nodes[0], n)

    group_of, members = {}, {}
    for rec in records:
        nodes = _group_nodes(rec)
        if nodes:
            g = find(nodes[0])
            group_of[rec["_key"]] = g
    for n in parent:
        members.setdefault(find(n), set()).add(n)
    return group_of, members, ungrouped


def split_groups(members, test_fraction, seed):
    """{group_id: 'test' | 'train_reserved'}. A group is held out if any of
    its board/session ids hashes below test_fraction."""
    return {
        g: "test" if any(_unit_hash(seed, n) < test_fraction for n in nodes) else "train_reserved"
        for g, nodes in members.items()
    }


def check_split_purity(records, group_of, side_of_group):
    """Leakage guard: no board or session id may appear on both sides."""
    sides = {}
    for rec in records:
        g = group_of.get(rec["_key"])
        if g is None:
            continue
        for n in _group_nodes(rec):
            sides.setdefault(n, set()).add(side_of_group[g])
    return [f"{n} appears in both the held-out and reserved splits" for n, s in sorted(sides.items()) if len(s) > 1]


def check_trained_on(test_records, trained_on):
    """Leakage guard against a model's own training manifest."""
    if not trained_on:
        return []
    caps = set(trained_on.get("capture_ids") or [])
    boards = set(trained_on.get("board_ids") or [])
    sessions = set(trained_on.get("session_ids") or [])
    problems = []
    for rec in test_records:
        if rec["capture_id"] in caps:
            problems.append(f"{rec['_key']}: held-out capture is in the model's training manifest")
        if rec.get("board_id") in boards:
            problems.append(f"{rec['_key']}: held-out board {rec['board_id']!r} is in the training manifest")
        if rec.get("session_id") in sessions:
            problems.append(f"{rec['_key']}: held-out session {rec['session_id']!r} is in the training manifest")
    return problems


def _pixel_digest(img):
    return hashlib.sha256(img.mode.encode() + repr(img.size).encode() + img.tobytes()).hexdigest()


def _dhash(img):
    small = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


# ─────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────

def _load_image(rec):
    with Image.open(rec["image_path"]) as im:
        return im.convert("RGB")


def evaluate_detection(test_records, images, group_of, detector, cfg, known_classes=None):
    recs = [r for r in test_records if r["kind"] == "detect"]
    if not recs:
        return {"status": "no_labeled_test_frames"}
    per_frame, classes = [], set()
    for rec in recs:
        preds = [p for p in detector(images[rec["_key"]]) if float(p.get("score", 1.0)) >= cfg.score_threshold]
        counts = match_frame(rec["ground_truth"]["boxes"], preds, cfg.iou_threshold)
        per_frame.append(counts)
        classes |= set(counts)
    classes = sorted(classes)
    rows = np.zeros((len(recs), len(classes), 3))
    for i, counts in enumerate(per_frame):
        for c, cls in enumerate(classes):
            rows[i, c] = counts.get(cls, (0, 0, 0))

    item_groups = [group_of[r["_key"]] for r in recs]
    resampler = Resampler(item_groups, cfg, np.random.default_rng(cfg.bootstrap_seed))
    boot = None
    if resampler.weights is not None:
        flat = rows.reshape(len(recs), -1)
        boot = (resampler.weights @ flat).reshape(-1, len(classes), 3)

    def block(tp, fp, fn, boot_tp=None, boot_fp=None, boot_fn=None):
        out = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "n_gt": int(tp + fn), "n_pred": int(tp + fp)}
        for name, num, den, bnum, bden in (
            ("precision", tp, tp + fp, boot_tp, None if boot_tp is None else boot_tp + boot_fp),
            ("recall", tp, tp + fn, boot_tp, None if boot_tp is None else boot_tp + boot_fn),
        ):
            if den == 0:
                out[name] = {"value": None, "note": "undefined: no predictions" if name == "precision" else "undefined: no ground-truth boxes"}
                continue
            reps = None
            if bnum is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    reps = bnum / bden
            out[name] = {
                "value": _r(num / den),
                "ci95_wilson_per_box": _interval(wilson_interval(num, den)),
                "ci95_bootstrap": _interval(resampler.ci(reps)),
                "bootstrap_method": resampler.method,
            }
        return out

    totals = rows.sum(axis=0)
    per_class = {}
    for c, cls in enumerate(classes):
        b = (None, None, None) if boot is None else (boot[:, c, 0], boot[:, c, 1], boot[:, c, 2])
        per_class[cls] = block(*totals[c], *b)
    pooled = totals.sum(axis=0)
    pooled_boot = (None, None, None) if boot is None else tuple(boot.sum(axis=1)[:, k] for k in range(3))

    out = {
        "status": "measured",
        "n_frames": len(recs),
        "n_groups": resampler.n_groups,
        "pooled_all_classes": block(*pooled, *pooled_boot),
        "per_class": per_class,
    }
    if resampler.warning:
        out["ci_warning"] = resampler.warning
    if known_classes is not None:
        unknown = sorted({b["class"] for r in recs for b in r["ground_truth"]["boxes"]} - set(known_classes))
        if unknown:
            out["labeled_classes_outside_model_taxonomy"] = unknown
    return out


def _gt_position_mm(gt, px_per_mm):
    if _finite(gt.get("delta_x_mm")):
        return gt["delta_x_mm"], gt["delta_y_mm"]
    if _finite(gt.get("delta_x_px")) and px_per_mm:
        return gt["delta_x_px"] / px_per_mm, gt["delta_y_px"] / px_per_mm
    return None


def _scale_for(rec, cfg):
    gt = rec["ground_truth"]
    if _finite(gt.get("px_per_mm")):
        return float(gt["px_per_mm"]), "ground_truth"
    ctx = rec.get("capture_context") or {}
    if _finite(ctx.get("px_per_mm")) and ctx["px_per_mm"] > 0:
        return float(ctx["px_per_mm"]), "capture_context"
    if cfg.px_per_mm:
        return float(cfg.px_per_mm), "cli"
    return None, "none"


def evaluate_alignment(test_records, images, group_of, aligners, cfg):
    recs = [r for r in test_records if r["kind"] == "align"]
    if not recs:
        return {"status": "no_labeled_test_frames"}
    scales = [_scale_for(r, cfg) for r in recs]
    scale_sources = {}
    for _, src in scales:
        scale_sources[src] = scale_sources.get(src, 0) + 1

    out = {
        "status": "measured",
        "n_frames": len(recs),
        "n_groups": len({group_of[r["_key"]] for r in recs}),
        "angle_period_deg": cfg.angle_period_deg,
        "px_per_mm_sources": scale_sources,
        "methods": {},
    }
    per_method = {}
    for name, fn in aligners.items():
        if isinstance(fn, str):
            out["methods"][name] = {"status": "not_run", "reason": fn}
            continue
        angle_err, pos_err, ex, ey, groups_a, groups_p, frame_rows = [], [], [], [], [], [], {}
        answered = 0
        for rec, (scale, _) in zip(recs, scales):
            pred = fn(images[rec["_key"]])
            if not pred.get("available") or not _finite(pred.get("delta_theta_deg")):
                continue
            answered += 1
            gt = rec["ground_truth"]
            g = group_of[rec["_key"]]
            a = abs(angle_error_deg(pred["delta_theta_deg"], gt["delta_theta_deg"], cfg.angle_period_deg))
            angle_err.append(a)
            groups_a.append(g)
            row = {"angle": a, "group": g}
            gt_mm = _gt_position_mm(gt, scale)
            if gt_mm is not None and scale and _finite(pred.get("delta_x_px")):
                dx = pred["delta_x_px"] / scale - gt_mm[0]
                dy = pred["delta_y_px"] / scale - gt_mm[1]
                ex.append(dx)
                ey.append(dy)
                pos_err.append(math.hypot(dx, dy))
                groups_p.append(g)
                row["pos"] = math.hypot(dx, dy)
            frame_rows[rec["_key"]] = row
        per_method[name] = frame_rows

        rng = np.random.default_rng(cfg.bootstrap_seed)
        res_a = Resampler(groups_a, cfg, rng)
        res_p = Resampler(groups_p, cfg, rng)
        m = {
            "status": "measured",
            "coverage": {
                "answered": answered, "of": len(recs), "value": _r(answered / len(recs)),
                "ci95_wilson": _interval(wilson_interval(answered, len(recs))),
                "note": "errors below are over the frames this path answered",
            },
            "angle_abs_error_deg": summarize_errors(angle_err, groups_a, res_a),
            "position_error_mm": summarize_errors(pos_err, groups_p, res_p),
        }
        if pos_err:
            m["position_bias_mm"] = {"x": summarize_signed_mean(ex, res_p), "y": summarize_signed_mean(ey, res_p)}
        else:
            m["position_error_mm"]["note"] = ("not measured: no frame had both a position label and a px_per_mm "
                                              "scale (calibrate before capturing, or pass --px-per-mm with its source)")
        warnings = {r.warning for r in (res_a, res_p) if r.n_items and r.warning}
        if warnings:
            m["ci_warning"] = sorted(warnings)
        out["methods"][name] = m

    measured = [n for n in aligners if n in per_method]
    if len(measured) == 2:
        a_name, b_name = measured
        both = [k for k in per_method[a_name] if k in per_method[b_name]]
        diff = {"b_minus_a": f"{b_name} - {a_name}", "n_frames_both_answered": len(both)}
        for metric, label in (("angle", "angle_abs_error_deg_mean_diff"), ("pos", "position_error_mm_mean_diff")):
            keys = [k for k in both if metric in per_method[a_name][k] and metric in per_method[b_name][k]]
            vals = [per_method[b_name][k][metric] - per_method[a_name][k][metric] for k in keys]
            res = Resampler([per_method[a_name][k]["group"] for k in keys], cfg,
                            np.random.default_rng(cfg.bootstrap_seed))
            diff[label] = summarize_signed_mean(vals, res) if vals else None
        out["paired_difference"] = diff
    return out


def evaluate(capture_dir, detectors=None, aligners=None, cfg=None, trained_on=None,
             known_classes=None, models=None):
    """Runs the whole held-out evaluation and returns the report dict.

    detectors / aligners: {name: callable(PIL image) or str reason-not-run}.
      a detector returns [{"class", "box": [x1,y1,x2,y2], "score"}, ...];
      an aligner returns {"available", "delta_theta_deg", "delta_x_px",
      "delta_y_px"} in the frame's own pixels (image center = 0, x right,
      y down, angle positive clockwise on screen), like /nn/align.
    Raises EvalRefused instead of returning numbers the data can't support."""
    cfg = cfg or EvalConfig()
    capture_dir = Path(capture_dir)
    if not capture_dir.is_dir():
        raise FileNotFoundError(f"capture dir not found: {capture_dir}")
    if not 0.0 < cfg.test_fraction <= 1.0:
        raise ValueError("test_fraction must be in (0, 1]")
    if cfg.px_per_mm is not None and not cfg.px_per_mm_source:
        raise ValueError("px_per_mm needs px_per_mm_source: state where the scale came from")

    records = load_records(capture_dir)
    synthetic = [_is_synthetic(r) for r in records]
    if any(synthetic) and not all(synthetic):
        raise EvalRefused("synthetic and real captures are mixed in one capture dir",
                          [r["_key"] for r, s in zip(records, synthetic) if s])
    is_synthetic = bool(records) and all(synthetic)

    group_of, members, ungrouped = group_records(records)
    side_of_group = split_groups(members, cfg.test_fraction, cfg.split_seed)
    problems = check_split_purity(records, group_of, side_of_group)
    if problems:
        raise EvalRefused("board/session split leaks", problems)

    test = [r for r in records if r["_key"] in group_of and side_of_group[group_of[r["_key"]]] == "test"]
    test_eval = [r for r in test if r["kind"] in EVAL_KINDS]

    problems = check_trained_on(test, trained_on)
    if problems:
        raise EvalRefused("held-out data overlaps a model's training data", problems)

    problems = []
    for rec in test_eval:
        problems += check_label_provenance(rec)
    if problems:
        raise EvalRefused("held-out labels were not all made by a person", problems)

    for rec in test_eval:
        problems += check_ground_truth(rec)
        if not Path(rec["image_path"]).is_file():
            problems.append(f"{rec['_key']}: image file missing ({rec['image_path']})")
    if problems:
        raise EvalRefused("held-out ground truth is unusable", problems)

    # Same image under two groups = the group ids are wrong, so the split
    # can't be trusted. Near-duplicates across splits are only flagged:
    # two copies of one board design can legitimately look alike.
    images, digests, hashes = {}, {}, {}
    eval_keys = {r["_key"] for r in test_eval}
    for rec in records:
        if rec["_key"] not in group_of or not Path(rec["image_path"]).is_file():
            continue
        img = _load_image(rec)
        if rec["_key"] in eval_keys:
            images[rec["_key"]] = img
        digests.setdefault(_pixel_digest(img), set()).add(group_of[rec["_key"]])
        hashes[rec["_key"]] = _dhash(img)
    dupes = [f"one image is filed under groups {sorted(g)}" for g in digests.values() if len(g) > 1]
    if dupes:
        raise EvalRefused("identical frames under different boards/sessions", dupes)
    near = []
    test_keys = {r["_key"] for r in test}
    for tk in sorted(k for k in hashes if k in test_keys):
        for ok in sorted(k for k in hashes if k not in test_keys):
            if bin(hashes[tk] ^ hashes[ok]).count("1") <= cfg.near_duplicate_max_hamming:
                near.append([tk, ok])

    test_groups = sorted({group_of[r["_key"]] for r in test})
    report = {
        "report": "SERC vision held-out evaluation",
        "data_provenance": "SYNTHETIC_TEST_FIXTURE" if is_synthetic else "real_captures",
        "provenance_note": ("SYNTHETIC test fixture: these numbers check the harness and describe no "
                            "real board or camera" if is_synthetic else
                            "measured on real SERC captures labeled by people, held out by board/session"),
        "harness_version": HARNESS_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "capture_dir": str(capture_dir),
        "conditions": asdict(cfg),
        "models": models or {},
        "split": {
            "unit": "board/session group (frames sharing a board_id or session_id)",
            "n_labeled_frames": len(records),
            "n_groups": len(members),
            "n_test_groups": len(test_groups),
            "test_groups": test_groups,
            "n_test_frames_evaluated": len(test_eval),
            "n_reserved_frames": sum(1 for r in records if r["_key"] in group_of) - len(test),
            "n_excluded_no_board_or_session_id": len(ungrouped),
            "trained_on_manifest_checked": bool(trained_on),
            "cross_split_near_duplicates": {
                "max_hamming": cfg.near_duplicate_max_hamming, "count": len(near), "pairs": near[:20],
            },
        },
        "detection": {"status": "not_run", "reason": "no detectors given"},
        "alignment": {"status": "not_run", "reason": "no aligners given"},
    }
    if detectors:
        report["detection"] = {"status": "evaluated", "detectors": {}}
        for name, fn in detectors.items():
            report["detection"]["detectors"][name] = (
                {"status": "not_run", "reason": fn} if isinstance(fn, str)
                else evaluate_detection(test_eval, images, group_of, fn, cfg, known_classes))
    if aligners:
        report["alignment"] = evaluate_alignment(test_eval, images, group_of, aligners, cfg)
    return report


def split_manifest(capture_dir, cfg=None):
    """Which ids are held out and which are reserved for fine-tuning. A
    training run should take only 'train_reserved' and record what it used,
    so the next evaluation can pass that record as --trained-on."""
    cfg = cfg or EvalConfig()
    records = load_records(Path(capture_dir))
    group_of, members, ungrouped = group_records(records)
    side_of_group = split_groups(members, cfg.test_fraction, cfg.split_seed)
    out = {"split_seed": cfg.split_seed, "test_fraction": cfg.test_fraction,
           "test": {"board_ids": set(), "session_ids": set(), "capture_ids": set()},
           "train_reserved": {"board_ids": set(), "session_ids": set(), "capture_ids": set()},
           "unassigned_no_board_or_session_id": sorted(k.split("/", 1)[1] for k in ungrouped)}
    for rec in records:
        g = group_of.get(rec["_key"])
        if g is None:
            continue
        side = out[side_of_group[g]]
        side["capture_ids"].add(rec["capture_id"])
        if rec.get("board_id"):
            side["board_ids"].add(rec["board_id"])
        if rec.get("session_id"):
            side["session_ids"].add(rec["session_id"])
    for side in ("test", "train_reserved"):
        out[side] = {k: sorted(v) for k, v in out[side].items()}
    return out


# ─────────────────────────────────────────────────────────────────────────
# Production models (CLI only — tests pass their own callables)
# ─────────────────────────────────────────────────────────────────────────

def _file_info(path):
    p = Path(path)
    if not p.is_file():
        return {"path": str(p.name), "exists": False}
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": p.name, "exists": True, "bytes": p.stat().st_size, "sha256": h.hexdigest()}


def load_production_models(detector_names=("yolo_hybrid",)):
    """The same weights and code paths flask_server.py serves. Anything that
    can't load comes back as a reason string, reported as not_run."""
    from pcb_jepa_nn import CLASS_NAMES, load_model, load_yolo_detector
    import jepa_heads
    import vision_classical

    best, yolo_w, heads = REPO_ROOT / "best.pt", REPO_ROOT / "yolov8_pcb.pt", REPO_ROOT / jepa_heads.CHECKPOINT_PATH
    models = {"classifier": _file_info(best), "yolo": _file_info(yolo_w), "jepa_heads": _file_info(heads)}
    classifier, loaded = load_model(str(best), device="cpu")

    detectors = {}
    for name in detector_names:
        if not loaded:
            detectors[name] = "best.pt did not load; the classifier would be random init"
        elif name == "yolo_hybrid":
            det = load_yolo_detector(classifier, str(yolo_w))
            try:
                det._ensure_loaded()
                detectors[name] = lambda img, det=det: det.detect(img)["boxes"]
            except Exception as e:  # missing weights / ultralytics not installed
                detectors[name] = f"YOLO hybrid unavailable: {e}"
        elif name == "sliding_window":
            detectors[name] = lambda img: classifier.detect_boxes_sliding(img)["boxes"]
        else:
            detectors[name] = f"unknown detector {name!r}"

    aligners = {"classical_cv": lambda img: vision_classical.estimate_alignment(img)}
    align_cnn, align_head, _, meta = jepa_heads.load_heads(str(heads))
    if align_cnn is None:
        aligners["jepa_synthetic_trained"] = f"{heads.name} missing or failed to load"
    else:
        def jepa(img):
            theta, dx, dy = jepa_heads.predict_alignment(img, align_cnn, align_head)
            return {"available": True, "delta_theta_deg": theta, "delta_x_px": dx, "delta_y_px": dy}
        aligners["jepa_synthetic_trained"] = jepa
        models["jepa_heads"]["training_data"] = (meta or {}).get("training_data")
    return detectors, aligners, models, list(CLASS_NAMES)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def _fmt(block):
    if not block or block.get("value") is None:
        return "n/a"
    ci = block.get("ci95") or block.get("ci95_bootstrap") or block.get("ci95_wilson_per_box")
    return f"{block['value']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def print_summary(report):
    print("=" * 72)
    print(f" {report['report']}  ({report['data_provenance']})")
    print(f" {report['provenance_note']}")
    s = report["split"]
    print(f" held out: {s['n_test_frames_evaluated']} frames in {s['n_test_groups']} board/session groups"
          f" ({s['n_excluded_no_board_or_session_id']} frames without ids excluded)")
    print("-" * 72)
    det = report["detection"]
    for name, d in (det.get("detectors") or {}).items():
        if d["status"] != "measured":
            print(f" detect/{name}: {d['status']} {d.get('reason', '')}")
            continue
        p = d["pooled_all_classes"]
        print(f" detect/{name}: {d['n_frames']} frames, {p['n_gt']} boxes  "
              f"precision {_fmt(p['precision'])}  recall {_fmt(p['recall'])}")
    al = report["alignment"]
    if al.get("status") != "measured":
        print(f" align: {al.get('status')} {al.get('reason', '')}")
    for name, m in (al.get("methods") or {}).items():
        if m["status"] != "measured":
            print(f" align/{name}: {m['status']} {m.get('reason', '')}")
            continue
        ang, pos = m["angle_abs_error_deg"], m["position_error_mm"]
        print(f" align/{name}: answered {m['coverage']['answered']}/{m['coverage']['of']}  "
              f"|angle err| mean {_fmt(ang.get('mean'))} deg  position err mean {_fmt(pos.get('mean'))} mm")
    print(" (95% CIs in brackets; see the JSON for method, n and warnings)")
    print("=" * 72)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Held-out evaluation on human-labeled SERC captures.")
    ap.add_argument("--capture-dir", default=data_capture.DEFAULT_CAPTURE_DIR)
    ap.add_argument("--out", default="vision_eval_report.json")
    ap.add_argument("--test-fraction", type=float, default=EvalConfig.test_fraction)
    ap.add_argument("--split-seed", default=EvalConfig.split_seed)
    ap.add_argument("--iou", type=float, default=EvalConfig.iou_threshold)
    ap.add_argument("--score-threshold", type=float, default=EvalConfig.score_threshold)
    ap.add_argument("--angle-period", type=float, default=EvalConfig.angle_period_deg,
                    help="fold angle errors by this symmetry (360 = none; 180/90 for symmetric parts)")
    ap.add_argument("--px-per-mm", type=float, help="fallback scale for frames that recorded none")
    ap.add_argument("--px-per-mm-source", help="required with --px-per-mm: where that scale was measured")
    ap.add_argument("--detectors", default="yolo_hybrid", help="comma list: yolo_hybrid,sliding_window")
    ap.add_argument("--no-detection", action="store_true")
    ap.add_argument("--no-alignment", action="store_true")
    ap.add_argument("--trained-on", help="JSON with capture_ids/board_ids/session_ids a model was trained on")
    ap.add_argument("--write-split", help="also write the held-out/reserved id manifest here")
    ap.add_argument("--n-boot", type=int, default=EvalConfig.n_boot)
    args = ap.parse_args(argv)

    cfg = EvalConfig(test_fraction=args.test_fraction, split_seed=args.split_seed,
                     iou_threshold=args.iou, score_threshold=args.score_threshold,
                     angle_period_deg=args.angle_period, px_per_mm=args.px_per_mm,
                     px_per_mm_source=args.px_per_mm_source, n_boot=args.n_boot)
    trained_on = None
    if args.trained_on:
        with open(args.trained_on) as f:
            trained_on = json.load(f)

    names = [n.strip() for n in args.detectors.split(",") if n.strip()]
    detectors, aligners, models, classes = load_production_models(names)
    try:
        report = evaluate(args.capture_dir,
                          detectors=None if args.no_detection else detectors,
                          aligners=None if args.no_alignment else aligners,
                          cfg=cfg, trained_on=trained_on, known_classes=classes, models=models)
    except EvalRefused as e:
        print("REFUSED - no numbers reported: " + e.reason, file=sys.stderr)
        for p in e.problems:
            print("  - " + p, file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as e:
        print("error: " + str(e), file=sys.stderr)
        return 1

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    if args.write_split:
        with open(args.write_split, "w") as f:
            json.dump(split_manifest(args.capture_dir, cfg), f, indent=2)
    print_summary(report)
    print(f" report: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
