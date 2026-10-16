"""ISO 9283-style analysis: known bias, noise and backlash put into the data
must come back out as AP, RP and vAP."""
import numpy as np
import pytest

from metrology import iso9283 as iso
from metrology.scara import ArmGeometry


def test_poses_follow_the_standard_layout():
    p = iso.iso_poses(0, 0, 100, 50)
    assert np.allclose(p["P1"], [50, 25])
    assert np.allclose(p["P2"], [10, 5])        # 10% of the diagonal in from (0, 0)
    assert np.allclose(p["P4"], [90, 45])       # and from the opposite corner


def test_plan_has_lead_in_cycles_and_vap_groups():
    rows = iso.plan((0, 0, 62, 42), ArmGeometry(), cycles=30, vap_cycles=10)
    assert rows[0]["test"] == "lead-in" and rows[0]["pose"] == "P5"
    assert sum(r["test"] == "AP_RP" for r in rows) == 150
    assert sum(r["test"] == "vAP" for r in rows) == 3 * 4 * 10
    assert all(r["x_pre_mm"] for r in rows if r["test"] == "vAP")


def _rows(pose, cmd, pts, test="AP_RP", approach="sequence"):
    return [{"test": test, "cycle": i + 1, "pose": pose, "approach": approach,
             "x_cmd_mm": str(cmd[0]), "y_cmd_mm": str(cmd[1]), "x_pre_mm": "", "y_pre_mm": "",
             "q1_cmd_deg": "0", "q2_cmd_deg": "0",
             "x_meas_mm": "%.6f" % p[0], "y_meas_mm": "%.6f" % p[1], "synthetic": ""} for i, p in enumerate(pts)]


def test_known_bias_and_noise_come_back_as_ap_and_rp():
    sigma, bias, cmd = 0.02, np.array([0.10, -0.05]), np.array([31.0, 21.0])
    pts = cmd + bias + np.random.default_rng(7).normal(0, sigma, (400, 2))
    res = iso.analyze(_rows("P1", cmd, pts))
    p = res["poses"]["P1"]
    assert p["AP_mm"] == pytest.approx(np.linalg.norm(bias), abs=0.005)
    # 2D isotropic Gaussian: l is Rayleigh, so l_bar + 3 S_l = (1.2533 + 3 * 0.6551) sigma
    assert p["RP_mm"] == pytest.approx(3.2185 * sigma, rel=0.1)
    assert p["std_x_mm"] == pytest.approx(sigma, rel=0.15)
    lo, hi = p["AP_ci95_mm"]
    assert lo < p["AP_mm"] < hi


def test_direction_dependent_offset_shows_up_as_vap():
    cmd = np.array([31.0, 21.0])
    rng = np.random.default_rng(1)
    rows = _rows("P1", cmd, cmd + [0.05, 0] + rng.normal(0, 0.005, (10, 2)), "vAP", "+x") + \
        _rows("P1", cmd, cmd + [-0.05, 0] + rng.normal(0, 0.005, (10, 2)), "vAP", "-x")
    v = iso.analyze(rows)["vap"]["P1"]
    assert v["vAP_mm"] == pytest.approx(0.10, abs=0.01)
    assert set(v["between"]) == {"+x", "-x"}


def test_simulated_backlash_shows_in_vap_not_in_rp():
    rows = iso.plan((0, 0, 62, 42), ArmGeometry(), cycles=20, vap_cycles=8)
    made_up = ArmGeometry((190.2, 39.9), 150.1, 179.9, (0.02, -0.01), (1.0, 1.0))
    res = iso.analyze(iso.simulate(rows, made_up, ArmGeometry(), noise_mm=0.01, backlash_deg=(0.1, 0.1)))
    worst_rp = max(p["RP_mm"] for p in res["poses"].values())
    assert min(v["vAP_mm"] for v in res["vap"].values()) > 2 * worst_rp
    assert res["synthetic"]
    assert "SYNTHETIC DATA" in iso.to_markdown(res)


def test_report_without_conditions_says_so():
    cmd = np.array([31.0, 21.0])
    md = iso.to_markdown(iso.analyze(_rows("P1", cmd, cmd + np.zeros((5, 2)))))
    assert "not recorded" in md


def test_no_measurements_is_an_error():
    with pytest.raises(ValueError):
        iso.analyze(iso.plan((0, 0, 62, 42), ArmGeometry(), cycles=1, vap_cycles=1))
