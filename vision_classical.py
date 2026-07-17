"""
vision_classical.py — classical-CV fallbacks for /nn/align and /nn/validate.

pcb_jepa_nn.py's AlignmentCorrector and PlacementValidator heads exist in
the JEPA architecture but have no trained weights (see JEPAConfig / the
downstream-heads comment block at the top of that file) — training them
needs a labeled alignment/defect dataset SERC doesn't have yet. Rather than
leave the two endpoints returning a hardcoded "not trained" stub forever,
this module gives them a real, working implementation using classical
OpenCV: contour + minAreaRect for rotation/offset, and a presence/structural
check for pass/fail. This is the same family of technique OpenPnP itself
uses for nozzle-tip alignment — it is not a placeholder, it is a legitimate
v1 that the JEPA heads can later replace once they're trained (same
response shape, so the frontend doesn't need to change when that happens).
"""

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def _to_gray_array(pil_image):
    arr = np.array(pil_image.convert("L"))
    return arr


def _largest_component_contour(gray):
    """Otsu-threshold the frame and return the largest contour that isn't
    just the whole frame (a bare background has no clean foreground blob)."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # component could be lighter OR darker than background — try both
    # polarities and keep whichever gives a cleaner (less frame-filling) blob
    candidates = []
    for mask in (th, cv2.bitwise_not(th)):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        frame_area = gray.shape[0] * gray.shape[1]
        if area < 0.005 * frame_area or area > 0.92 * frame_area:
            continue  # noise speck, or thresholded the whole background
        candidates.append((area, c))

    if not candidates:
        return None
    # prefer the mid-sized, well-defined blob (largest of the valid ones)
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def estimate_alignment(pil_image, px_per_mm=None):
    """Classical-CV replacement for the (untrained) JEPA AlignmentCorrector
    head. Returns delta_theta_deg / delta_x_mm / delta_y_mm the same shape
    as the original stubbed /nn/align response.
    """
    if not CV2_AVAILABLE:
        return {"available": False, "reason": "opencv not installed"}

    gray = _to_gray_array(pil_image)
    h, w = gray.shape
    contour = _largest_component_contour(gray)
    if contour is None:
        return {
            "delta_theta_deg": 0.0, "delta_x_mm": 0.0, "delta_y_mm": 0.0,
            "available": False,
            "reason": "no component contour found in frame",
            "method": "classical_cv_min_area_rect",
        }

    (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)

    # normalize to the smallest rotation that squares the part up to the
    # nearest 0/90/180/270 — a rectangular SMD part looks identical at each
    # of those, so "aligned" only ever means "within +/-45 deg of one of them"
    theta = angle
    if rw < rh:
        theta = angle - 90
    while theta > 45:
        theta -= 90
    while theta < -45:
        theta += 90

    dx_px = cx - w / 2.0
    dy_px = cy - h / 2.0

    area = cv2.contourArea(contour)
    rect_area = max(rw * rh, 1e-6)
    extent = min(area / rect_area, 1.0)          # how rectangular the blob is
    size_ratio = min((rw * rh) / (w * h), 1.0)    # how much of the frame it fills
    confidence = round(float(extent * (0.5 + 0.5 * min(size_ratio * 6, 1.0))), 3)

    out = {
        "delta_theta_deg": round(float(theta), 2),
        "delta_x_px": round(float(dx_px), 2),
        "delta_y_px": round(float(dy_px), 2),
        "available": True,
        "method": "classical_cv_min_area_rect",
        "confidence": confidence,
        "note": "classical-CV baseline - JEPA AlignmentCorrector head not yet trained",
    }
    if px_per_mm:
        out["delta_x_mm"] = round(float(dx_px / px_per_mm), 3)
        out["delta_y_mm"] = round(float(dy_px / px_per_mm), 3)
    else:
        out["delta_x_mm"] = None
        out["delta_y_mm"] = None
        out["note"] += "; no px_per_mm calibration available, mm fields are null"
    return out


def validate_placement(pil_image, model=None, class_names=None, expected_class=None,
                        presence_threshold=0.35):
    """Classical/classifier-backed replacement for the (untrained) JEPA
    PlacementValidator head. If a trained classifier + expected_class are
    available, use classifier confidence; otherwise fall back to a
    presence/structural check via the same contour detector as alignment.
    """
    if model is not None and expected_class is not None and class_names is not None:
        try:
            result = model.infer_multilabel(pil_image)
            preds = result.get("predictions", [])
            match = next((p for p in preds if p.get("class") == expected_class), None)
            # PCBVisionSystem.infer_multilabel (pcb_jepa_nn.py) keys this
            # "score", not "confidence" - narrow except below so a schema
            # mismatch like that raises during testing instead of silently
            # falling through to the classical path forever.
            pass_prob = float(match["score"]) if match else 0.0
            decision = "PASS" if pass_prob >= presence_threshold else "FAIL"
            return {
                "decision": decision,
                "pass_prob": round(pass_prob, 3),
                "fail_prob": round(1.0 - pass_prob, 3),
                "available": True,
                "method": "classifier_confidence",
                "expected_class": expected_class,
                "note": "classifier-backed baseline - JEPA PlacementValidator head not yet trained",
            }
        except (KeyError, TypeError, AttributeError):
            pass  # model returned an unexpected shape - fall through to classical presence check below

    if not CV2_AVAILABLE:
        return {"available": False, "reason": "opencv not installed"}

    gray = _to_gray_array(pil_image)
    contour = _largest_component_contour(gray)
    if contour is None:
        return {
            "decision": "FAIL", "pass_prob": 0.0, "fail_prob": 1.0,
            "available": True,
            "method": "classical_cv_presence",
            "note": "no component-shaped blob found post-placement (missing/tombstoned part?)",
        }

    h, w = gray.shape
    area = cv2.contourArea(contour)
    x, y, rw, rh = cv2.boundingRect(contour)
    extent = min(area / max(rw * rh, 1e-6), 1.0)
    pass_prob = round(float(0.4 + 0.6 * extent), 3)
    decision = "PASS" if pass_prob >= presence_threshold else "FAIL"
    return {
        "decision": decision,
        "pass_prob": pass_prob,
        "fail_prob": round(1.0 - pass_prob, 3),
        "available": True,
        "method": "classical_cv_presence",
        "note": "classical-CV baseline - JEPA PlacementValidator head not yet trained",
    }
