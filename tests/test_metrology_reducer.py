"""Reducer incoming inspection (metrology/reducer_inspect.py): injected
transmission error, lost motion and stiffness come back out, the gate is
honest about noise, and only the strain-wave option in the budget changes."""
import json
import shutil

import cv2
import numpy as np
import pytest

from metrology import charuco_calib as cc
from metrology import error_budget as eb
from metrology import reducer_inspect as ri


def test_plan_steps_one_way_with_repeated_photos():
    rows = ri.plan_te(frames=3)
    measured = [r for r in rows if r["sweep"] in ("coarse", "fine")]
    assert len(measured) == 3 * (80 + 64) and len(rows) == len(measured) + 2
    steps = [r["input_microsteps"] for r in rows]
    assert steps == sorted(steps)                       # never reverses, so lost motion stays out
    assert len({r["image"] for r in measured}) == len(measured)


def test_injected_te_components_are_recovered():
    res = ri.analyze(ri.simulate_te(ri.plan_te(), te_out_arcmin=1.5, te_in_arcmin=0.4))
    assert res["coarse"]["amplitudes_arcmin"][0] == pytest.approx(1.5, abs=0.1)
    assert res["fine"]["amplitudes_arcmin"][1] == pytest.approx(0.4, abs=0.05)     # twice per input turn
    assert res["te_half_pp_arcmin"] == pytest.approx(1.9, abs=0.15)
    lo, hi = res["te_half_pp_ci95_arcmin"]
    assert lo < res["te_half_pp_arcmin"] < hi
    assert res["synthetic"]


@pytest.mark.parametrize("te_out, expected", [(0.5, "pass"), (3.0, "fail"), (1.5, "uncertain")])
def test_gate_passes_fails_or_says_uncertain(te_out, expected):
    res = ri.analyze(ri.simulate_te(ri.plan_te(), te_out_arcmin=te_out, te_in_arcmin=0.4))
    assert res["te_verdict"] == expected


def test_noisy_measurement_is_not_judged():
    res = ri.analyze(ri.simulate_te(ri.plan_te(frames=1), te_out_arcmin=0.5, noise_arcmin=3.0))
    assert res["noise_limited"] and res["te_verdict"] == "noisy"
    assert "too noisy" in ri.to_markdown(res, "x")


def test_target_reading_backwards_gives_the_same_te():
    rows = ri.simulate_te(ri.plan_te(), te_out_arcmin=1.0)
    flipped = [dict(r, angle_deg="%.6f" % -float(r["angle_deg"])) if r["angle_deg"] else r for r in rows]
    assert ri.analyze(flipped)["te_half_pp_arcmin"] == pytest.approx(ri.analyze(rows)["te_half_pp_arcmin"], abs=1e-6)


def test_lost_motion_and_stiffness_from_angles_or_a_dial():
    rows = ri.simulate_lost_motion(ri.plan_lost_motion(), lost_motion_arcmin=1.5, stiffness=2200.0)
    lm = ri.lost_motion(rows)
    assert lm["lost_motion_arcmin"] == pytest.approx(1.5, abs=0.15)
    assert lm["stiffness_nm_per_rad"] == pytest.approx(2200.0, rel=0.1)
    dial = [dict(r, angle_deg="", dial_mm="%.6f" % (np.radians(float(r["angle_deg"])) * 50.0)) for r in rows]
    assert ri.lost_motion(dial, radius_mm=50.0)["lost_motion_arcmin"] == pytest.approx(lm["lost_motion_arcmin"], abs=1e-3)
    with pytest.raises(ValueError):
        ri.lost_motion(dial)                                   # dial readings need the radius


def test_budget_update_touches_only_the_strain_wave_option(tmp_path):
    cfg = tmp_path / "budget_config.json"
    shutil.copy(eb.DEFAULT_CONFIG, cfg)
    res = ri.analyze(ri.simulate_te(ri.plan_te(), te_out_arcmin=1.0),
                     ri.simulate_lost_motion(ri.plan_lost_motion()))
    gate = ri.apply_to_budget(res, cfg, "SYNTHETIC test")
    before = {s["name"]: s for s in json.loads(eb.DEFAULT_CONFIG.read_text(encoding="utf-8"))["scenarios"]}
    after = {s["name"]: s for s in json.loads(cfg.read_text(encoding="utf-8"))["scenarios"]}
    belt = next(n for n in before if n.startswith("Option B"))
    sw = next(n for n in before if n.startswith("Option D"))
    assert after[belt] == before[belt]
    te = after[sw]["set"]["joints.j1.reducer_transmission_error_deg"]
    assert te["source"].startswith("measured") and te["value"] == pytest.approx(res["te_half_pp_arcmin"] / 60, abs=1e-4)
    assert gate["scenario"] == sw and "accuracy_rms_mm" in gate


def test_synthetic_readings_never_go_into_the_real_config(tmp_path):
    te = tmp_path / "te.csv"
    ri.main(["simulate", "--out", str(te)])
    original = eb.DEFAULT_CONFIG.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        ri.main(["analyze", str(te), "--update-config"])
    assert eb.DEFAULT_CONFIG.read_text(encoding="utf-8") == original


def test_camera_chain_reads_te_from_rendered_photos():
    """A target on the output, rendered by a made-up camera with lens
    distortion at 24 positions with 3 arcmin of once-per-turn TE."""
    board = cc.make_board((6, 6), 6.0, 4.5, "DICT_4X4_50")
    img = cc.board_image(board, 10.0)
    tilt, _ = cv2.Rodrigues(np.array([0.08, -0.05, 0.0]))
    cmd = np.arange(24) * 15.0
    shots = []
    for i, c in enumerate(cmd):
        turn, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(c + 3.0 / 60 * np.sin(np.radians(c) + 0.4))]))
        R = tilt @ turn
        t = np.array([0, 0, 110.0]) - R @ np.array([18.0, 18.0, 0])
        shots.append(cc.render_view(img, 10.0, cc.DEMO_K, cc.DEMO_DIST, cv2.Rodrigues(R)[0], t, (640, 480), seed=i))
    intr = {"camera_matrix": cc.DEMO_K.tolist(), "dist_coeffs": cc.DEMO_DIST.tolist()}
    angles = ri.signed_angles(ri.board_rotations(board, intr, shots))
    rows = [{"sweep": "coarse", "input_microsteps": str(i), "output_cmd_deg": "%.4f" % c, "angle_deg": "%.6f" % a}
            for i, (c, a) in enumerate(zip(cmd, angles))]
    res = ri.analyze(rows)
    assert res["coarse"]["amplitudes_arcmin"][0] == pytest.approx(3.0, abs=0.75)
