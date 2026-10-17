"""Error budget (metrology/error_budget.py): the numbers follow from the
inputs, every input is traceable, and the scenarios move the right way."""
import json
import math

import numpy as np
import pytest

from metrology import error_budget as eb
from metrology.scara import ArmGeometry, fit_geometry, forward, plan_calibration_poses


@pytest.fixture(scope="module")
def default_report():
    return eb.run()


def _terms(scenario):
    return {t["id"]: t for t in scenario["terms"]}


def test_every_term_says_where_its_input_came_from(default_report):
    for sc in default_report["scenarios"]:
        for t in sc["terms"]:
            assert t["source"].strip(), t["id"]
            assert t["measure"].strip() and t["fix"].strip(), t["id"]


def test_step_resolution_matches_a_hand_calculation(default_report):
    sc = default_report["scenarios"][0]
    micro_deg = 360.0 / (200 * 16)                          # direct drive, 16 microsteps
    expected = math.radians(micro_deg / math.sqrt(12)) * sc["worst_point_radius_mm"]
    assert _terms(sc)["j1_step_quant"]["tip_rms_mm"] == pytest.approx(expected, rel=1e-3)


def test_shares_add_up_and_terms_are_ranked(default_report):
    for sc in default_report["scenarios"]:
        assert sum(t["share"] for t in sc["terms"]) == pytest.approx(1.0)
        tips = [t["tip_rms_mm"] for t in sc["terms"]]
        assert tips == sorted(tips, reverse=True)


def _scenario(report, prefix):
    return next(sc for sc in report["scenarios"] if sc["name"].startswith(prefix))


def test_each_step_lowers_error_and_options_branch_from_the_last_step(default_report):
    ladder = [sc for sc in default_report["scenarios"] if not sc["option"]]
    acc = [sc["accuracy_rms_mm"] for sc in ladder]
    assert all(b < a for a, b in zip(acc, acc[1:])), acc
    options = [sc for sc in default_report["scenarios"] if sc["option"]]
    assert len(options) == 4 and all(o["accuracy_rms_mm"] < acc[-1] for o in options)
    recommended = [o for o in options if o["recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["accuracy_rms_mm"] == min(o["accuracy_rms_mm"] for o in options)


def test_reduction_divides_motor_side_errors_and_stiffens_the_joint(default_report):
    direct = _terms(default_report["scenarios"][2])
    reduced = _terms(_scenario(default_report, "Option D"))
    assert reduced["j1_step_accuracy"]["input_sigma"] == pytest.approx(direct["j1_step_accuracy"]["input_sigma"] / 50)
    assert reduced["j1_load_deflection"]["input_sigma"] < direct["j1_load_deflection"]["input_sigma"] / 100
    assert "j1_reducer_te" in reduced and "j1_reducer_te" not in direct


def test_output_side_encoder_is_not_divided_by_the_ratio(default_report):
    on_output = _terms(_scenario(default_report, "Option C"))      # belt 4:1, encoder on the joint
    direct = _terms(_scenario(default_report, "Option A"))         # direct drive, encoder on the motor
    assert on_output["j1_enc_inl"]["input_sigma"] == pytest.approx(direct["j1_enc_inl"]["input_sigma"])
    assert "j1_reducer_te" not in on_output and "j1_load_deflection" not in on_output   # the loop corrects them


def test_calibration_term_is_predicted_until_measured(default_report):
    fit = _terms(default_report["scenarios"][1])["kinematic_fit"]
    assert fit["source"].startswith("predicted for 25 planned poses")
    assert 0 < fit["tip_rms_mm"] < 0.05


def test_one_sided_approach_takes_backlash_out_of_repeatability(default_report):
    calibrated, one_sided = default_report["scenarios"][1], default_report["scenarios"][2]
    assert one_sided["repeatability_std_mm"] < 0.2 * calibrated["repeatability_std_mm"]


def test_closed_loop_replaces_open_loop_stepper_terms(default_report):
    ids = _terms(default_report["scenarios"][3])
    assert "j1_enc_inl" in ids and "j2_enc_inl" in ids
    assert "j1_step_accuracy" not in ids and "j1_load_deflection" not in ids
    assert "j2_homing" not in ids   # an absolute encoder sets the zero


def test_command_line_override_changes_the_result():
    fine = eb.run(overrides=[("joints.j1.microsteps", 256)])
    coarse = eb.run()
    assert _terms(fine["scenarios"][0])["j1_step_quant"]["tip_rms_mm"] < \
        _terms(coarse["scenarios"][0])["j1_step_quant"]["tip_rms_mm"] / 10


def test_markdown_report_flags_it_as_a_prediction(default_report):
    md = eb.to_markdown(default_report)
    assert "Predicted, not measured" in md
    assert "## Assumptions to replace with measurements" in md
    assert "`joints.j1.load_torque_nm`" in md


def test_measured_calibration_feeds_its_covariance_into_the_budget(tmp_path):
    true = ArmGeometry((188.0, 42.0), 150.6, 179.4, (0.7, -0.4), (1.0, 1.0))
    _, q = plan_calibration_poses(ArmGeometry(), (62, 42), n=25)
    fit = fit_geometry(q, forward(true, q) + np.random.default_rng(3).normal(0, 0.02, q.shape))
    path = tmp_path / "arm_calibration.json"
    path.write_text(json.dumps({"created": "test", "synthetic": True, "geometry": fit.geometry.to_dict(),
                                "fit": fit.summary()}))
    rep = eb.run(arm_calibration=str(path))
    calibrated = _terms(rep["scenarios"][1])
    assert "kinematic_fit" in calibrated and "j1_zero" not in calibrated and "base_registration" not in calibrated
    assert 0 < calibrated["kinematic_fit"]["tip_rms_mm"] < 0.1
    assert rep["synthetic_calibration"]
    assert "SYNTHETIC calibration file" in eb.to_markdown(rep)
