import cv2
import numpy as np
from PIL import Image

import vision_classical as vc


def _make_rect_image(size=200, center=(120, 90), wh=(60, 30), angle_deg=15, bg=255, fg=0):
    canvas = np.full((size, size), bg, dtype=np.uint8)
    box = cv2.boxPoints(((center[0], center[1]), wh, angle_deg)).astype(np.int32)
    cv2.fillPoly(canvas, [box], fg)
    return Image.fromarray(canvas)


def test_alignment_recovers_known_rotation_and_offset():
    img = _make_rect_image(center=(120, 90), wh=(60, 30), angle_deg=15)
    res = vc.estimate_alignment(img, px_per_mm=10.0)
    assert res["available"]
    assert abs(res["delta_x_px"] - 20) < 2
    assert abs(res["delta_y_px"] - (-10)) < 2
    assert min(abs(res["delta_theta_deg"] - 15), abs(res["delta_theta_deg"] + 75)) < 3
    assert res["delta_x_mm"] == round(res["delta_x_px"] / 10.0, 3)


def test_centered_part_has_near_zero_correction():
    img = _make_rect_image(center=(100, 100), wh=(50, 50), angle_deg=0)
    res = vc.estimate_alignment(img, px_per_mm=None)
    assert abs(res["delta_x_px"]) < 2
    assert abs(res["delta_y_px"]) < 2
    assert res["delta_x_mm"] is None  # no px_per_mm supplied


def test_blank_frame_is_unavailable():
    blank = Image.fromarray(np.full((200, 200), 255, dtype=np.uint8))
    res = vc.estimate_alignment(blank)
    assert res["available"] is False


def test_validate_presence_pass_and_fail():
    present = _make_rect_image(center=(100, 100), wh=(60, 40), angle_deg=5)
    blank = Image.fromarray(np.full((200, 200), 255, dtype=np.uint8))
    assert vc.validate_placement(present)["decision"] == "PASS"
    assert vc.validate_placement(blank)["decision"] == "FAIL"


def test_validate_classifier_path_uses_score_key():
    """Regression test: infer_multilabel keys predictions 'score', not
    'confidence' — this broke silently once (fell through to the classical
    path on a KeyError) until the schema mismatch was caught in integration
    testing. Pin the real key name down here so it can't regress quietly."""
    present = _make_rect_image(center=(100, 100), wh=(60, 40), angle_deg=5)

    class FakeModel:
        def infer_multilabel(self, pil_img):
            return {"predictions": [{"class": "0603_resistor", "score": 0.91},
                                     {"class": "0805_capacitor", "score": 0.04}]}

    hit = vc.validate_placement(present, model=FakeModel(), class_names=["0603_resistor"], expected_class="0603_resistor")
    assert hit["method"] == "classifier_confidence"
    assert hit["decision"] == "PASS"

    miss = vc.validate_placement(present, model=FakeModel(), class_names=["0805_capacitor"], expected_class="0805_capacitor")
    assert miss["decision"] == "FAIL"
