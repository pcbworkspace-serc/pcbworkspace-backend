"""End-to-end tests against the real Flask app (test client) — real image
bytes through /nn/align and /nn/validate, real IK/DRC/placement/routing
math, Claude calls mocked (no ANTHROPIC_API_KEY in CI) via monkeypatching
flask_server._claude_complete.
"""
import io
import json

import cv2
import numpy as np
from PIL import Image


def _make_rect_png_bytes(center=(120, 90), wh=(60, 30), angle_deg=15):
    canvas = np.full((200, 200), 255, dtype=np.uint8)
    box = cv2.boxPoints((center, wh, angle_deg)).astype(np.int32)
    cv2.fillPoly(canvas, [box], 0)
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_and_status(client):
    assert client.get("/health").status_code == 200
    assert client.get("/nn/status").status_code == 200


def test_calibration_save_get_roundtrip_with_board_size(app, client):
    pixel_pts = [[0, 0], [400, 0], [400, 300], [0, 300]]
    world_pts = [[0, 0], [100, 0], [100, 75], [0, 75]]
    r = client.post("/calibration/save", json={
        "pixel_points": pixel_pts, "world_points": world_pts,
        "board_size_mm": {"w_mm": 100, "h_mm": 75},
    })
    assert r.status_code == 200
    assert r.get_json()["board_size_mm"] == {"w_mm": 100, "h_mm": 75}

    assert client.get("/calibration/get").get_json()["board_size_mm"] == {"w_mm": 100, "h_mm": 75}

    bw, bh = app._get_board_bounds()
    assert (bw, bh) == (100.0, 75.0)

    client.post("/calibration/clear")  # don't leak calibration.json state into other tests


def test_vla_plan_clamps_hallucinated_actions(app, client, monkeypatch):
    def fake_claude(system, messages, max_tokens=1024):
        return json.dumps({
            "interpretation": "test move",
            "actions": [
                {"action": "home"},
                {"action": "move", "x_mm": 31, "y_mm": 21, "z_mm": 5},
                {"action": "move", "x_mm": 99999, "y_mm": 21, "z_mm": 5},  # hallucinated
                {"action": "nuke_the_board"},                              # unknown type
                {"action": "pick"},
            ],
            "warnings": [],
        })
    monkeypatch.setattr(app, "_claude_complete", fake_claude)

    r = client.post("/vla/plan", data={"instruction": "pick up R1", "board_state": "[]"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert len(body["actions"]) == 3
    assert body["dropped_action_count"] == 2
    assert len(body["warnings"]) >= 2


def test_vla_plan_drops_in_bounds_but_kinematically_unreachable_target(app, client, monkeypatch):
    """A target can pass the board-bounds clamp and still be outside what
    the arm's own 2-link kinematics can reach - e.g. a corner of an
    oversized calibrated board. filter_unreachable_actions is the second
    safety net that catches that case."""
    client.post("/calibration/save", json={
        "pixel_points": [[0, 0], [400, 0], [400, 300], [0, 300]],
        "world_points": [[0, 0], [500, 0], [500, 500], [0, 500]],
        "board_size_mm": {"w_mm": 500, "h_mm": 500},
    })
    try:
        def fake_claude(system, messages, max_tokens=1024):
            return json.dumps({
                "interpretation": "reach corner",
                "actions": [{"action": "move", "x_mm": 500, "y_mm": 500, "z_mm": 5}],  # in-bounds, unreachable
                "warnings": [],
            })
        monkeypatch.setattr(app, "_claude_complete", fake_claude)

        r = client.post("/vla/plan", data={"instruction": "go to far corner", "board_state": "[]"})
        body = r.get_json()
        assert body["actions"] == []
        assert body["dropped_action_count"] == 1
        assert any("outside arm reach envelope" in w for w in body["warnings"])
    finally:
        client.post("/calibration/clear")


def test_schematic_generate_and_drc(app, client, monkeypatch):
    def fake_claude(system, messages, max_tokens=2048):
        return json.dumps({
            "interpretation": "LED current-limiting resistor",
            "circuit": {"components": [
                {"ref": "D1", "type": "led", "footprint": "0805", "pins": {"1": "VIN", "2": "MID"}},
                {"ref": "R1", "type": "resistor", "footprint": "0603", "pins": {"1": "MID", "2": "GND"}},
            ]},
        })
    monkeypatch.setattr(app, "_claude_complete", fake_claude)

    r = client.post("/schematic/generate", json={"description": "LED with current limiting resistor"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert not body["drc"]["clean"]  # dangling VIN/GND, no connector in this circuit
    # the mocked circuit's R1 has no "value" field - bom_check correctly flags that
    assert "bom_check" in body and body["bom_check"]["plausible"] is False
    assert any("no value specified" in w for w in body["bom_check"]["warnings"])

    good_circuit = {"components": [
        {"ref": "J1", "type": "connector", "pins": {"1": "VIN", "2": "GND"}},
        {"ref": "D1", "type": "led", "footprint": "0805", "pins": {"1": "VIN", "2": "MID"}},
        {"ref": "R1", "type": "resistor", "footprint": "0603", "pins": {"1": "MID", "2": "GND"}},
    ]}
    r = client.post("/schematic/drc", json={"circuit": good_circuit})
    assert r.get_json()["drc"]["clean"] is True

    r = client.post("/schematic/bom_check", json={"circuit": {"components": [
        {"ref": "R1", "type": "resistor", "value": "100nF", "pins": {"1": "A", "2": "B"}},  # cap value on a resistor
    ]}})
    assert r.status_code == 200
    body = r.get_json()
    assert body["bom_check"]["plausible"] is False
    assert any("doesn't look like a resistance" in w for w in body["bom_check"]["warnings"])

    r = client.post("/schematic/export/kicad", json={"circuit": good_circuit})
    assert r.status_code == 200
    assert r.mimetype == "text/plain"
    text = r.get_data(as_text=True)
    assert text.startswith('(export (version "D")')
    assert '(ref "R1")' in text and '(name "MID")' in text
    assert "layla_export.net" in r.headers.get("Content-Disposition", "")


def test_layout_place_and_route(client):
    comps = [
        {"ref": "J1", "type": "connector", "pins": {"1": "VIN", "2": "GND"}},
        {"ref": "D1", "type": "led", "footprint": "0805", "pins": {"1": "VIN", "2": "MID"}},
        {"ref": "R1", "type": "resistor", "footprint": "0603", "pins": {"1": "MID", "2": "GND"}},
    ]
    r = client.post("/layout/place", json={"components": comps, "board_w_mm": 40, "board_h_mm": 30,
                                            "iterations": 800, "seed": 3})
    assert r.status_code == 200
    positions = r.get_json()["positions"]
    assert set(positions.keys()) == {"J1", "D1", "R1"}

    r = client.post("/layout/route", json={"components": comps, "positions": positions,
                                            "board_w_mm": 40, "board_h_mm": 30})
    assert r.status_code == 200
    route_body = r.get_json()
    assert len(route_body["routed_nets"]) + len(route_body["unrouted_nets"]) == 3

    r = client.post("/layout/route", json={"components": comps, "board_w_mm": 40, "board_h_mm": 30})
    assert r.status_code == 200 and "placement" in r.get_json()  # auto-places when no positions given


def test_nn_align_uses_trained_head_when_available(app, client):
    r = client.post("/nn/align", data={"image": (_make_rect_png_bytes(), "test.png")},
                     content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["available"] is True
    # method depends on whether jepa_heads_synthetic.pt is present in this environment
    assert body["method"] in ("trained_head_synthetic", "classical_cv_min_area_rect")


def test_nn_align_falls_back_to_classical_cv_when_checkpoint_missing(app, client, monkeypatch):
    monkeypatch.setattr(app, "_jepa_heads_load_attempted", False)
    monkeypatch.setattr(app, "_align_cnn", None)
    monkeypatch.setattr(app, "_align_head", None)
    monkeypatch.setattr(app, "_validate_head", None)
    monkeypatch.setattr(app.jepa_heads, "load_heads", lambda path=None: (None, None, None, None))

    r = client.post("/nn/align", data={"image": (_make_rect_png_bytes(), "test.png")},
                     content_type="multipart/form-data")
    assert r.get_json()["method"] == "classical_cv_min_area_rect"


def test_nn_validate_expected_class_uses_classifier_path(client):
    r = client.post("/nn/validate", data={
        "image": (_make_rect_png_bytes(center=(100, 100), wh=(50, 50), angle_deg=0), "test.png"),
        "expected_class": "R",
    }, content_type="multipart/form-data")
    assert r.status_code == 200
    assert r.get_json()["method"] == "classifier_confidence"
