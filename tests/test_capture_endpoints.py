"""End-to-end test of the capture-then-label loop through the real
endpoints: /nn/align (or /validate) -> capture_id -> /nn/feedback ->
/nn/capture_stats reflects it."""
import io

import cv2
import numpy as np
from PIL import Image


def _make_rect_png_bytes():
    canvas = np.full((200, 200), 255, dtype=np.uint8)
    box = cv2.boxPoints(((100, 100), (60, 30), 10)).astype(np.int32)
    cv2.fillPoly(canvas, [box], 0)
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_capture_disabled_by_default_no_capture_id(client):
    r = client.post("/nn/align", data={"image": (_make_rect_png_bytes(), "t.png")},
                     content_type="multipart/form-data")
    assert "capture_id" not in r.get_json()


def test_capture_and_feedback_roundtrip(app, client, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "CAPTURE_TRAINING_DATA", True)
    monkeypatch.setattr(app, "CAPTURE_DIR", str(tmp_path))

    r = client.post("/nn/align", data={"image": (_make_rect_png_bytes(), "t.png")},
                     content_type="multipart/form-data")
    body = r.get_json()
    assert "capture_id" in body
    capture_id = body["capture_id"]

    r = client.post("/nn/feedback", json={
        "kind": "align", "capture_id": capture_id,
        "ground_truth": {"delta_theta_deg": 9.5, "delta_x_mm": 0.1, "delta_y_mm": -0.2},
    })
    assert r.status_code == 200
    assert r.get_json()["capture"]["ground_truth"]["delta_theta_deg"] == 9.5

    stats = client.get("/nn/capture_stats").get_json()
    assert stats["capture_enabled"] is True
    assert stats["stats"]["align"] == {"total_captured": 1, "labeled": 1, "unlabeled": 0}


def test_feedback_unknown_capture_id_404(app, client, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "CAPTURE_DIR", str(tmp_path))
    r = client.post("/nn/feedback", json={"kind": "align", "capture_id": "nope", "ground_truth": {"x": 1}})
    assert r.status_code == 404


def test_feedback_missing_fields_400(client):
    r = client.post("/nn/feedback", json={"kind": "align"})
    assert r.status_code == 400


def test_capture_records_board_session_and_human_labeler(app, client, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "CAPTURE_TRAINING_DATA", True)
    monkeypatch.setattr(app, "CAPTURE_DIR", str(tmp_path))

    r = client.post("/nn/align", data={"image": (_make_rect_png_bytes(), "t.png"),
                                       "board_id": "B1", "session_id": "S1"},
                    content_type="multipart/form-data")
    capture_id = r.get_json()["capture_id"]

    r = client.post("/nn/feedback", json={
        "kind": "align", "capture_id": capture_id, "ground_truth": {"delta_theta_deg": 1.0},
        "label_source": "human",
    })
    assert r.status_code == 400  # a human label has to say who made it

    r = client.post("/nn/feedback", json={
        "kind": "align", "capture_id": capture_id, "ground_truth": {"delta_theta_deg": 1.0},
        "label_source": "human", "labeler": "jd", "board_id": "B2",
    })
    assert r.status_code == 400  # can't move a capture to another board after the fact

    r = client.post("/nn/feedback", json={
        "kind": "align", "capture_id": capture_id, "ground_truth": {"delta_theta_deg": 1.0},
        "label_source": "human", "labeler": "jd",
    })
    cap = r.get_json()["capture"]
    assert (cap["board_id"], cap["session_id"], cap["label_source"], cap["labeler"]) == ("B1", "S1", "human", "jd")
    assert cap["capture_context"]["image_size"] == [200, 200]


def test_detect_endpoint_captures_for_box_labeling(app, client, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "CAPTURE_TRAINING_DATA", True)
    monkeypatch.setattr(app, "CAPTURE_DIR", str(tmp_path))
    r = client.post("/nn/detect_boxes", data={"image": (_make_rect_png_bytes(), "t.png"), "board_id": "B1"},
                    content_type="multipart/form-data")
    capture_id = r.get_json()["capture_id"]
    r = client.post("/nn/feedback", json={
        "kind": "detect", "capture_id": capture_id,
        "ground_truth": {"boxes": [{"class": "R", "box": [70, 85, 130, 115]}]},
        "label_source": "human", "labeler": "jd",
    })
    assert r.status_code == 200
    assert client.get("/nn/capture_stats").get_json()["stats"]["detect"]["labeled"] == 1
