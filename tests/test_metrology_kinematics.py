"""Kinematic model + geometry fit (metrology/scara.py) and the layout_engine
hook that swaps the ARM_BASE_OFFSET_MM placeholder for measured geometry."""
import importlib
import json

import numpy as np
import pytest

import layout_engine
from metrology.scara import ArmGeometry, fit_geometry, forward, inverse, plan_calibration_poses

TRUE = ArmGeometry((183.2, 47.9), 151.1, 178.6, (-23.5, 1.3), (1.004, 0.997))


def _board_grid():
    return np.array([(x, y) for x in np.linspace(0, 62, 7) for y in np.linspace(0, 42, 5)])


@pytest.mark.parametrize("elbow", [1, -1])
def test_inverse_then_forward_returns_the_same_board_point(elbow):
    pts = _board_grid()
    q = inverse(TRUE, pts, elbow=elbow)
    assert not np.isnan(q).any()
    assert np.abs(forward(TRUE, q) - pts).max() < 1e-9


def test_unreachable_points_come_back_nan():
    q = inverse(ArmGeometry(), np.array([[1000.0, 0.0], [10.0, 10.0]]))
    assert np.isnan(q[0]).all() and not np.isnan(q[1]).any()


def test_fit_recovers_a_rotated_mounted_arm_from_noisy_measurements():
    """The arm here is mounted 23.5 deg rotated with off-nominal links,
    base and steps-per-degree. Measurement noise 0.02 mm."""
    _, q = plan_calibration_poses(ArmGeometry(), (62, 42), n=25)
    meas = forward(TRUE, q) + np.random.default_rng(1).normal(0, 0.02, q.shape)
    r = fit_geometry(q, meas, initial=ArmGeometry())
    g = r.geometry
    assert g.l1_mm > 0 and g.l2_mm > 0
    assert abs(g.zero_deg[0] - TRUE.zero_deg[0]) < 0.5
    # what matters for placement: predicted tip position across the board
    q_board = inverse(TRUE, _board_grid())
    assert np.abs(forward(g, q_board) - forward(TRUE, q_board)).max() < 0.05
    assert 0.01 < r.rms_mm < 0.04
    assert r.cv_rms_mm < 2 * r.rms_mm
    assert not r.mirrored_fits_better
    assert set(r.std_error) == set(r.fitted) and r.covariance.shape == (len(r.fitted), len(r.fitted))


def test_fit_warns_when_measurements_are_in_y_down_image_coordinates():
    _, q = plan_calibration_poses(ArmGeometry(), (62, 42), n=25)
    meas = forward(TRUE, q) * np.array([1.0, -1.0])
    assert fit_geometry(q, meas).mirrored_fits_better


def test_fit_refuses_too_few_poses():
    _, q = plan_calibration_poses(ArmGeometry(), (62, 42), n=25)
    with pytest.raises(ValueError):
        fit_geometry(q[:6], forward(TRUE, q[:6]))


def _record(tmp_path, synthetic):
    path = tmp_path / "arm_calibration.json"
    path.write_text(json.dumps({"created": "2026-10-16T12:00:00", "synthetic": synthetic,
                                "geometry": {"base_offset_mm": [150.0, 30.0], "l1_mm": 151.0, "l2_mm": 179.0}}))
    return path


@pytest.mark.parametrize("synthetic", [False, True])
def test_layout_engine_uses_measured_geometry_but_never_a_synthetic_one(tmp_path, monkeypatch, synthetic):
    monkeypatch.setenv("ARM_CALIBRATION_PATH", str(_record(tmp_path, synthetic)))
    try:
        le = importlib.reload(layout_engine)
        if synthetic:
            assert le.ARM_GEOMETRY_SOURCE == "placeholder"
            assert le.ARM_BASE_OFFSET_MM == (190.0, 40.0)
        else:
            assert le.ARM_GEOMETRY_SOURCE.startswith("measured")
            assert le.ARM_BASE_OFFSET_MM == (150.0, 30.0)
            assert le.check_reach(0, 0)[1] == pytest.approx(np.hypot(150.0, 30.0))   # defaults use measured values
            _, warnings = le.filter_unreachable_actions([{"action": "move", "x_mm": 900, "y_mm": 0}])
            assert "measured" in warnings[0] and "placeholder" not in warnings[0]
    finally:
        monkeypatch.delenv("ARM_CALIBRATION_PATH")
        importlib.reload(layout_engine)
