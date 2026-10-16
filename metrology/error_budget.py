"""
error_budget.py — placement error budget for the MiniMEE arm.

Every error source (step resolution and accuracy, load deflection, backlash,
encoder non-linearity, link lengths, zero offsets, camera calibration,
nozzle runout, release slip...) is turned into a tip error in mm using the
arm's own Jacobian, at the WORST point on the board, then combined and
compared with the team's own targets (hardware_plan_v3, section 9).

It is a design-stage prediction. Inputs come from metrology/budget_config.json,
where each number says whether it is a datasheet value, a hardware-doc
value or an assumption; the report repeats that. Measurements replace
assumptions; the ISO 9283 test (iso9283.py) is what checks the prediction.

Error classes:
  random     changes every cycle         -> repeatability and accuracy
  direction  depends on approach side    -> repeatability too, unless every
                                            target is approached one way
  fixed      same every time at a pose   -> accuracy only
  session    changes per homing / board  -> accuracy only

Systematic terms are combined like random ones (variance sum) across the
board, i.e. the result is an RMS over many placements, which is how the
team's 0.15 mm RMS target is phrased.

    python -m metrology.error_budget                       # markdown report to stdout
    python -m metrology.error_budget --out metrology/ERROR_BUDGET.md --json budget.json
    python -m metrology.error_budget --set joints.j1.microsteps=32
    python -m metrology.error_budget --arm-calibration arm_calibration.json
"""

import argparse
import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from metrology.scara import (ArmGeometry, inverse, local_directions, plan_calibration_poses,
                             predicted_fit_covariance, tip_covariance)

DEFAULT_CONFIG = Path(__file__).with_name("budget_config.json")
SQRT3, SQRT12 = math.sqrt(3.0), math.sqrt(12.0)


# ─────────────────────────────────────────────────────────────────────────
# Config handling: leaves are either plain values or {"value", "source"}
# ─────────────────────────────────────────────────────────────────────────

def _is_leaf(node):
    return isinstance(node, dict) and "value" in node


def unwrap(node):
    if _is_leaf(node):
        return node["value"]
    if isinstance(node, dict):
        return {k: unwrap(v) for k, v in node.items() if not k.startswith("_")}
    if isinstance(node, list):
        return [unwrap(v) for v in node]
    return node


def source_of(raw, dotted):
    node = raw
    for part in dotted.split("."):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return node.get("source", "") if _is_leaf(node) else ""


def set_value(raw, dotted, value, source=None):
    parts = dotted.split(".")
    node = raw
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    leaf = parts[-1]
    if _is_leaf(node.get(leaf)):
        node[leaf]["value"] = value
        if source:
            node[leaf]["source"] = source
    else:
        node[leaf] = {"value": value, "source": source or "set on the command line"}


def _parse_cli_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# ─────────────────────────────────────────────────────────────────────────
# Terms
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Term:
    id: str
    label: str
    kind: str        # 'joint1' | 'joint2' (sigma in deg) | 'link1' | 'link2' | 'radial' | 'xy' (sigma in mm)
    sigma: float     # 1-sigma; deg for joint kinds, mm otherwise (per axis for 'xy')
    cls: str         # random | direction | fixed | session
    source: str
    measure: str
    fix: str
    extra: dict = field(default_factory=dict)


def _joint_terms(c, raw, j):
    """Joint-angle error terms, in degrees at the joint (output side).

    Reduction ratio N divides everything that happens on the motor side
    (step size, step error, motor-shaft encoder) and multiplies the motor's
    holding stiffness by N^2. The reducer adds its own stiffness (in series),
    lost motion and transmission error on the output side. Closed loop on a
    motor-shaft encoder corrects the motor angle only; closed loop on an
    output-side encoder corrects everything up to that encoder's accuracy."""
    J = c["joints"][j]
    enc = J["encoder"]
    n = float(J["gear_ratio"])
    full_step = 360.0 / J["full_steps_per_rev"]               # motor degrees
    micro = full_step / J["microsteps"] / n                     # joint degrees
    kind = "joint1" if j == "j1" else "joint2"
    lab = J["label"]
    src = lambda key: source_of(raw, "joints.%s.%s" % (j, key))
    calibrated = c["calibration"]["kinematic"]
    one_sided = c["approach"] == "unidirectional"
    residual = c["unidirectional_residual_frac"] if (one_sided and calibrated) else 1.0
    closed = enc["present"] and enc["closed_loop"]
    at_output = enc.get("location", "motor") == "output"
    enc_div = 1.0 if at_output else n
    k_red = float(J.get("reducer_stiffness_nm_per_rad") or 0.0)        # 0 = rigid
    k_motor = J["holding_torque_nm"] * J["hold_current_frac"] * (math.pi / 2) / math.radians(full_step)
    terms = []

    if closed:
        inl = enc["linearized_residual_deg"] if enc["linearized"] else enc["inl_deg"]
        count = 360.0 / 2 ** enc["bits"] / enc_div
        terms += [
            Term(j + "_enc_inl", "%s encoder non-linearity (%s shaft)" % (lab, "output" if at_output else "motor"),
                 kind, inl / enc_div / SQRT3, "fixed",
                 src("encoder.linearized_residual_deg" if enc["linearized"] else "encoder.inl_deg"),
                 "Rotate the joint through its range against a reference (camera on the ChArUco grid, or "
                 "averaged full-step positions) and log encoder angle vs reference angle.",
                 "Per-unit lookup-table linearization in firmware; centre the magnet within the datasheet limits."),
            Term(j + "_enc_quant", "%s encoder resolution" % lab, kind, count / SQRT12, "fixed",
                 src("encoder.bits"), "Fixed by the encoder's bit count.", "Reduction between joint and encoder shaft."),
            Term(j + "_deadband", "%s closed-loop settling band" % lab, kind, max(micro, count) / SQRT3, "random",
                 src("microsteps"), "Log encoder error at standstill after each move.",
                 "Finer microstepping, or settle on the encoder count rather than the step count."),
        ]
        stiffness = None if at_output else (k_red or None)
    else:
        terms += [
            Term(j + "_step_quant", "%s step resolution" % lab, kind, micro / SQRT12, "fixed",
                 src("microsteps"), "Fixed by steps/rev, microsteps and gear ratio.",
                 "Reduction between motor and joint."),
            Term(j + "_step_accuracy", "%s stepper position error (+/-%g%% of a full step)" % (lab, J["step_accuracy_pct"]),
                 kind, J["step_accuracy_pct"] / 100.0 * full_step / n / SQRT3, "fixed", src("step_accuracy_pct"),
                 "Step through 4 full steps in microsteps and read the real angle (encoder or camera); repeat at a few angles.",
                 "Closed loop on a linearized encoder, or reduction."),
        ]
        k_joint = k_motor * n * n
        stiffness = 1.0 / (1.0 / k_joint + 1.0 / k_red) if k_red else k_joint

    if stiffness:
        terms.append(Term(j + "_load_deflection", "%s load deflection (hose drag, friction)" % lab, kind,
                          math.degrees(J["load_torque_nm"] / stiffness) * residual, "direction", src("load_torque_nm"),
                          "Push the nozzle tangentially with a force gauge at standstill and read the angle change; "
                          "hang the hose and measure its pull.",
                          "Reduction (motor stiffness grows with ratio squared), a stiffer reducer, closed loop, lighter hose routing."))

    if not (closed and at_output):
        if J["backlash_deg"]:
            terms.append(Term(j + "_backlash", "%s backlash in the coupling" % lab, kind,
                              J["backlash_deg"] / 2.0 * residual, "direction", src("backlash_deg"),
                              "Approach the same target from both sides and compare (the ISO test's vAP does this).",
                              "Approach every target from one side; clamp or bolt the link to the shaft or flange."))
        if J.get("reducer_backlash_deg"):
            terms.append(Term(j + "_reducer_lost_motion", "%s reducer lost motion (%s)" % (lab, J.get("reducer", "reducer")),
                              kind, J["reducer_backlash_deg"] / 2.0 * residual, "direction", src("reducer_backlash_deg"),
                              "Lock the motor, torque the output both ways at a few % of rated torque, read the angle.",
                              "Approach every target from one side."))
        if J.get("reducer_transmission_error_deg"):
            terms.append(Term(j + "_reducer_te", "%s reducer transmission error (%s)" % (lab, J.get("reducer", "reducer")),
                              kind, J["reducer_transmission_error_deg"] / SQRT3, "fixed", src("reducer_transmission_error_deg"),
                              "Turn the input in steps over a full output revolution against an output-side reference "
                              "(camera on a target on the link) and log the error.",
                              "Per-joint error map from that measurement, or an output-side reference encoder."))

    if not calibrated:
        terms.append(Term(j + "_zero", "%s zero offset" % lab, kind, J["zero_uncertainty_deg"] / SQRT3, "fixed",
                          src("zero_uncertainty_deg"),
                          "Kinematic fit (metrology/kinematic_fit.py) estimates it with a standard error.",
                          "Kinematic calibration."))
    return terms + _homing_term(J, j, kind, lab, src)


def _homing_term(J, j, kind, lab, src):
    if J["encoder"]["present"]:
        return []   # an absolute encoder sets zero; its error is in the encoder terms
    return [Term(j + "_homing", "%s homing repeatability" % lab, kind,
                 J["homing_repeatability_deg"] / SQRT3, "session", src("homing_repeatability_deg"),
                 "Home 20 times and read the joint angle at a fixed pose each time.",
                 "Absolute encoder on the joint, or a better home switch.")]


def build_terms(c, raw):
    terms = _joint_terms(c, raw, "j1") + _joint_terms(c, raw, "j2")
    L, cal = c["links"], c["calibration"]
    ct, cb, nz, pr, pk = c["camera_top"], c["camera_bottom"], c["nozzle"], c["process"], c["package"]

    if cal["kinematic"]:
        if cal.get("fit"):
            names, cov, fsrc = cal["fit"]["names"], np.asarray(cal["fit"]["cov"], dtype=float), source_of(raw, "calibration.fit")
        else:
            # Not measured yet: the covariance the planned fit would have.
            g = c["geometry"]
            geom = ArmGeometry(tuple(g["base_offset_mm"]), g["l1_mm"], g["l2_mm"],
                               tuple(g.get("zero_deg", (0.0, 0.0))), tuple(g.get("scale", (1.0, 1.0))))
            _, q = plan_calibration_poses(geom, tuple(g["board_mm"]), n=cal["planned_poses"],
                                          margin_mm=cal["pose_margin_mm"], elbow=g["elbow"])
            names, cov = predicted_fit_covariance(geom, q, cal["tip_measurement_sigma_mm"])
            fsrc = "predicted for %d planned poses; tip measurement %s" % (
                len(q), source_of(raw, "calibration.tip_measurement_sigma_mm"))
        terms.append(Term("kinematic_fit", "Kinematic calibration uncertainty (zeros, links, base offset)", "fitcov", 0.0,
                          "fixed", fsrc, "The fit's parameter covariance, carried to each board point.",
                          "More, wider-spread calibration poses; a better tip measurement.",
                          extra={"names": names, "cov": cov}))
    else:
        for k in ("link1", "link2"):
            terms.append(Term(k + "_length", "%s length" % k.replace("link", "Link "), k, L["length_tolerance_mm"] / SQRT3,
                              "fixed", source_of(raw, "links.length_tolerance_mm"),
                              "Kinematic fit; or calipers between joint axes.", "Kinematic calibration."))
        terms.append(Term("base_registration", "Arm base position vs board (placeholder offset)", "xy",
                          c["registration"]["base_offset_uncertainty_mm"] / SQRT3, "fixed",
                          source_of(raw, "registration.base_offset_uncertainty_mm"),
                          "Kinematic fit against board-frame measurements.",
                          "Kinematic calibration replaces ARM_BASE_OFFSET_MM."))

    # Link bending under the hose side load: cantilever F L^3 / 3EI per tube,
    # plus the printed joints. The tube part is small; it's listed so the
    # report shows it was checked rather than ignored.
    o, wall = L["tube_outer_mm"], L["tube_wall_mm"]
    inner = max(0.0, o - 2 * wall)
    I = (o ** 4 - inner ** 4) / 12.0
    E = L["tube_modulus_gpa"] * 1000.0
    F = L["lateral_load_n"]
    bend = sum(F * ln ** 3 / (3 * E * I) for ln in (c["geometry"]["l1_mm"], c["geometry"]["l2_mm"]))
    terms.append(Term("compliance", "Link and printed-joint bending under hose load", "xy",
                      (bend + L["printed_joint_compliance_mm"]) / SQRT3, "fixed",
                      "tube: computed F*L^3/(3EI) = %.4f mm; joints: %s" % (bend, source_of(raw, "links.printed_joint_compliance_mm")),
                      "Dial indicator on the nozzle while pulling sideways with a known force.",
                      "Stiffer printed joints (more walls, metal inserts), hose strain relief."))
    thermal = L["petg_cte_ppm_per_k"] * 1e-6 * L["printed_length_mm"] * L["temperature_swing_k"]
    terms.append(Term("thermal", "Thermal growth of printed parts", "radial", thermal / SQRT3, "session",
                      source_of(raw, "links.petg_cte_ppm_per_k"),
                      "Repeat the ISO test cold and after 30 minutes of running.", "Warm-up before calibrating and placing."))

    half_field = 0.5 * math.hypot(*c["geometry"]["board_mm"])
    terms.append(Term("fiducials", "Top camera fiducial location", "xy",
                      ct["fiducial_sigma_px"] / ct["px_per_mm"] / math.sqrt(ct["fiducials_per_board"]), "session",
                      source_of(raw, "camera_top.fiducial_sigma_px"),
                      "Locate the same fiducials in 30 frames without moving anything; take the std.",
                      "More fiducials, better lighting, higher resolution."))
    if cal["camera"]:
        terms.append(Term("top_distortion", "Top camera residual after ChArUco calibration", "xy",
                          ct["reprojection_rms_px"] / ct["px_per_mm"] / math.sqrt(2), "fixed",
                          source_of(raw, "camera_top.reprojection_rms_px"),
                          "charuco_calib.py reports reprojection RMS (and held-out RMS).", "More views, flatter target."))
        plane = ct["plane_offset_after_mm"]
        psrc = source_of(raw, "camera_top.plane_offset_after_mm")
    else:
        terms.append(Term("top_distortion", "Top camera lens distortion (uncorrected)", "xy",
                          ct["distortion_edge_pct"] / 100.0 * half_field / math.sqrt(6), "fixed",
                          source_of(raw, "camera_top.distortion_edge_pct"),
                          "charuco_calib.py measures it.", "ChArUco intrinsics + distortion before the homography."))
        plane = ct["plane_offset_mm"]
        psrc = source_of(raw, "camera_top.plane_offset_mm")
    terms.append(Term("top_parallax", "Top camera height mismatch (homography plane vs board top)", "xy",
                      plane * half_field / ct["camera_height_mm"] / math.sqrt(6), "fixed", psrc,
                      "Put a 1.6 mm shim under the target and see how far the computed points move.",
                      "Take the homography with the target at board-top height."))

    terms.append(Term("bottom_locate", "Bottom camera part location", "xy", cb["sigma_px"] / cb["px_per_mm"], "random",
                      source_of(raw, "camera_bottom.sigma_px"),
                      "Measure the same held part 30 times without moving it.", "Lighting, resolution."))
    scale = cb["scale_error_after_pct"] if cal["camera"] else cb["scale_error_pct"]
    terms.append(Term("bottom_scale", "Bottom camera mm-per-pixel error", "xy",
                      scale / 100.0 * cb["pick_offset_mm"] / SQRT3, "random",
                      source_of(raw, "camera_bottom.scale_error_after_pct" if cal["camera"] else "camera_bottom.scale_error_pct"),
                      "Image a feature of known size at the nozzle height.", "Calibrate the bottom camera scale."))
    nozzle_c = (cb["sigma_px"] / cb["px_per_mm"]) if cal["nozzle"] else cb["nozzle_center_uncertainty_mm"] / SQRT3
    terms.append(Term("nozzle_center", "Nozzle axis position in the bottom image", "xy", nozzle_c, "fixed",
                      source_of(raw, "camera_bottom.sigma_px" if cal["nozzle"] else "camera_bottom.nozzle_center_uncertainty_mm"),
                      "Rotate the bare nozzle over the bottom camera and fit a circle.", "Nozzle-tip calibration."))
    runout = nz["runout_after_mm"] if cal["nozzle"] else nz["runout_tir_mm"] / 2.0
    terms.append(Term("nozzle_runout", "Nozzle runout", "xy", runout / math.sqrt(2), "random",
                      source_of(raw, "nozzle.runout_after_mm" if cal["nozzle"] else "nozzle.runout_tir_mm"),
                      "Same circle fit; runout is the circle diameter.", "Runout compensation (OpenPnP has one)."))
    wr = c["wrist"]
    wrist_q = 360.0 / wr["full_steps_per_rev"] / wr["microsteps"] / SQRT12
    rot = math.radians(math.hypot(cb["rotation_sigma_deg"], wrist_q))
    terms.append(Term("rotation", "Part rotation error at the part's end (%s)" % pk["name"], "xy",
                      rot * pk["length_mm"] / 2.0, "random", source_of(raw, "camera_bottom.rotation_sigma_deg"),
                      "Measure placed-part angle under a microscope.", "Better angle estimate; finer wrist steps."))
    terms.append(Term("release", "Part shift on vacuum release", "xy", pr["release_slip_mm"], "random",
                      source_of(raw, "process.release_slip_mm"),
                      "Place onto tacky tape, compare the part before and after release with the top camera.",
                      "Lower placement height, controlled blow-off."))
    terms.append(Term("settle", "Residual vibration at placement", "xy", pr["settle_amplitude_mm"] / math.sqrt(2), "random",
                      source_of(raw, "process.settle_amplitude_mm"),
                      "High-frame-rate top camera on the nozzle after a move.", "Longer settle, TPU mounts, slower last approach."))
    return terms


# ─────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────

def _term_cov(term, dirs, i):
    if term.kind == "fitcov":
        return tip_covariance(dirs["geom"], dirs["q"][i:i + 1], term.extra["names"], term.extra["cov"])[0]
    if term.kind in ("joint1", "joint2"):
        v = dirs["d_theta1" if term.kind == "joint1" else "d_theta2"][i] * math.radians(term.sigma)
        return np.outer(v, v)
    if term.kind in ("link1", "link2", "radial"):
        u = dirs["u_" + term.kind][i] * term.sigma
        return np.outer(u, u)
    return np.eye(2) * term.sigma ** 2


def _radial_stats(cov, n=200000, seed=0):
    """95th-percentile radial error, and the ISO 9283 RP statistic
    (mean distance + 3 std of distance) for zero-mean 2D Gaussian errors."""
    if not np.any(cov):
        return 0.0, 0.0
    s = np.random.default_rng(seed).multivariate_normal([0.0, 0.0], cov, n, method="eigh")
    r = np.linalg.norm(s, axis=1)
    return float(np.percentile(r, 95)), float(r.mean() + 3 * r.std(ddof=1))


def board_grid(geom_cfg, nx=9, ny=7):
    w, h = geom_cfg["board_mm"]
    return np.array([(x, y) for y in np.linspace(0, h, ny) for x in np.linspace(0, w, nx)])


def evaluate(c, terms):
    g = c["geometry"]
    geom = ArmGeometry(tuple(g["base_offset_mm"]), g["l1_mm"], g["l2_mm"],
                       tuple(g.get("zero_deg", (0.0, 0.0))), tuple(g.get("scale", (1.0, 1.0))))
    pts = board_grid(g)
    q = inverse(geom, pts, elbow=g["elbow"])
    if np.isnan(q).any():
        raise ValueError("part of the board is outside the arm's reach with this geometry")
    dirs = local_directions(geom, q)
    dirs["geom"], dirs["q"] = geom, q
    repeat_classes = {"random"} | ({"direction"} if c["approach"] == "bidirectional" else set())

    covs = np.array([[_term_cov(t, dirs, i) for i in range(len(pts))] for t in terms])   # (T,P,2,2)
    acc = covs.sum(axis=0)
    worst = int(np.argmax(np.trace(acc, axis1=1, axis2=2)))
    rep_mask = np.array([t.cls in repeat_classes for t in terms])
    rep = covs[rep_mask].sum(axis=0) if rep_mask.any() else np.zeros_like(acc)

    per_term = []
    total_var = float(np.trace(acc[worst]))
    for t, cv in zip(terms, covs):
        var = float(np.trace(cv[worst]))
        per_term.append({"id": t.id, "label": t.label, "class": t.cls,
                         "input_sigma": t.sigma,
                         "input_unit": "deg" if t.kind.startswith("joint") else ("fit covariance" if t.kind == "fitcov" else "mm"),
                         "tip_rms_mm": math.sqrt(var), "share": var / total_var if total_var else 0.0,
                         "source": t.source, "measure": t.measure, "fix": t.fix})
    per_term.sort(key=lambda d: -d["tip_rms_mm"])

    acc95, _ = _radial_stats(acc[worst])
    _, rp = _radial_stats(rep[worst])
    rms_all = np.sqrt(np.trace(acc, axis1=1, axis2=2))
    target = c["target"]
    rep_std = float(math.sqrt(max(np.linalg.eigvalsh(rep[worst])))) if rep_mask.any() else 0.0
    return {
        "worst_point_board_mm": [round(float(v), 1) for v in pts[worst]],
        "worst_point_radius_mm": round(float(dirs["r_mm"][worst]), 1),
        "accuracy_rms_mm": math.sqrt(total_var),
        "accuracy_rms_board_mean_mm": float(np.sqrt(np.mean(rms_all ** 2))),
        "accuracy_95_radial_mm": acc95,
        "repeatability_std_mm": rep_std,
        "iso_rp_mm": rp,
        "meets_accuracy_target": math.sqrt(total_var) <= target["accuracy_rms_mm"],
        "meets_repeatability_target": rep_std <= target["repeatability_std_mm"],
        "terms": per_term,
    }


def distance_sweep(c, raw, radii=(100, 130, 160, 190, 220, 250, 280)):
    """Same scenario, board moved along its current direction from the base.
    J1's error at the tip scales with distance from the base, so this shows
    how much a closer board buys before any hardware change."""
    g = c["geometry"]
    w, h = g["board_mm"]
    t = np.array(g["base_offset_mm"], dtype=float)
    centre = t + [w / 2, h / 2]
    unit = centre / np.linalg.norm(centre)
    out = []
    for r in radii:
        cc = copy.deepcopy(c)
        cc["geometry"]["base_offset_mm"] = list(unit * r - [w / 2, h / 2])
        try:
            res = evaluate(cc, build_terms(cc, raw))
        except ValueError:
            continue
        out.append({"board_centre_radius_mm": r, "accuracy_rms_mm": res["accuracy_rms_mm"]})
    return out


def collect_assumed(node, prefix=""):
    """(input path, value, source) for every input whose source says 'assumed'."""
    out = []
    if _is_leaf(node):
        if "assumed" in node.get("source", "").lower():
            out.append((prefix, node["value"], node["source"]))
    elif isinstance(node, dict):
        for k, v in node.items():
            if not k.startswith("_") and k != "scenarios":
                out += collect_assumed(v, prefix + "." + k if prefix else k)
    return out


def run(config_path=DEFAULT_CONFIG, overrides=(), arm_calibration=None):
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for dotted, value in overrides:
        set_value(raw, dotted, value)
    if arm_calibration:
        cal = json.loads(Path(arm_calibration).read_text(encoding="utf-8"))
        g, fit = cal["geometry"], cal["fit"]
        label = "measured: %s (%s, %d poses)" % (Path(arm_calibration).name, cal.get("created", "undated"), fit["n_points"])
        if cal.get("synthetic"):
            label = "SYNTHETIC (made-up arm) " + label
        for key in ("l1_mm", "l2_mm", "base_offset_mm", "zero_deg", "scale"):
            set_value(raw, "geometry." + key, g[key], label)
        set_value(raw, "calibration.fit", {"names": fit["fitted_parameters"], "cov": fit["covariance"]}, label)

    results, states = [], {}
    cumulative = copy.deepcopy(raw)
    for sc in raw.get("scenarios") or [{"name": "Config as given", "set": {}}]:
        if sc.get("from"):
            cumulative = copy.deepcopy(states[sc["from"]])   # a design option branches off an earlier step
        for dotted, value in sc.get("set", {}).items():
            if _is_leaf(value):
                set_value(cumulative, dotted, value["value"], value.get("source") or "scenario: %s" % sc["name"])
            else:
                set_value(cumulative, dotted, value, "scenario: %s" % sc["name"])
        states[sc["name"]] = copy.deepcopy(cumulative)
        c = unwrap(cumulative)
        res = evaluate(c, build_terms(c, cumulative))
        res["name"] = sc["name"]
        res["hypothetical"] = bool(sc.get("hypothetical"))
        res["option"] = bool(sc.get("from"))
        res["recommended"] = bool(sc.get("recommended"))
        res["_cfg"], res["_raw"] = c, copy.deepcopy(cumulative)
        results.append(res)

    real = [r for r in results if not r["hypothetical"]]
    recommended = [r for r in results if r["recommended"]]
    best = recommended[0] if recommended else (real[-1] if real else results[-1])
    sweep = distance_sweep(best["_cfg"], best["_raw"])
    assumed = collect_assumed(raw)
    base_c = unwrap(raw)
    for r in results:
        r.pop("_cfg")
        r.pop("_raw")
    synthetic = bool(arm_calibration) and bool(json.loads(Path(arm_calibration).read_text(encoding="utf-8")).get("synthetic"))
    return {"target": base_c["target"], "package": base_c["package"], "scenarios": results, "synthetic_calibration": synthetic,
            "distance_sweep": {"scenario": best["name"], "rows": sweep}, "assumptions": assumed,
            "config": str(config_path), "arm_calibration": str(arm_calibration) if arm_calibration else None}


# ─────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────

def _f(x, nd=3):
    return ("%%.%df" % nd) % x


def to_markdown(rep):
    t = rep["target"]
    lines = ["# MiniMEE placement error budget", ""]
    if rep.get("synthetic_calibration"):
        lines += ["**SYNTHETIC calibration file.** The geometry and fit covariance below come from a made-up arm, "
                  "used to test the pipeline. They describe nothing real.", ""]
    lines += [
             "**Predicted, not measured.** Generated by `python -m metrology.error_budget` from `%s`%s. "
             "The arm has not been assembled, so every number below comes from datasheets, the hardware docs or a stated "
             "assumption. The ISO 9283 test (`metrology/iso9283.py`) on the real arm is what confirms or refutes it."
             % (Path(rep["config"]).name, " and %s" % Path(rep["arm_calibration"]).name if rep["arm_calibration"] else ""),
             "",
             "Targets (hardware_plan_v3, section 9): accuracy < %s mm RMS, repeatability < %s mm std. "
             "Errors are evaluated at the worst point on the board, for an %s." % (t["accuracy_rms_mm"], t["repeatability_std_mm"], rep["package"]["name"]),
             "", "## Scenarios", "",
             "Steps without a letter each add to the step above. Lettered options are alternative joint drives, "
             "each applied on top of the step it branches from.", "",
             "| Scenario | Accuracy RMS (mm) | 95% radial (mm) | Repeatability std (mm) | ISO RP (mm) | Meets targets |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rep["scenarios"]:
        ok = ("yes" if r["meets_accuracy_target"] else "no") + " / " + ("yes" if r["meets_repeatability_target"] else "no")
        lines.append("| %s%s%s | %s | %s | %s | %s | %s |" % (
            r["name"], " *(hypothetical)*" if r["hypothetical"] else "", " **(recommended)**" if r["recommended"] else "",
            _f(r["accuracy_rms_mm"]),
            _f(r["accuracy_95_radial_mm"]), _f(r["repeatability_std_mm"]), _f(r["iso_rp_mm"]), ok))
    lines += ["", "\"Meets targets\" is accuracy / repeatability. ISO RP = mean distance + 3 std of distance, from the "
              "cycle-to-cycle terms only.", ""]

    for r in rep["scenarios"]:
        lines += ["## %s" % r["name"], "",
                  "Worst board point (%s, %s) mm, %s mm from the base. Board-average accuracy RMS: %s mm." % (
                      r["worst_point_board_mm"][0], r["worst_point_board_mm"][1], r["worst_point_radius_mm"],
                      _f(r["accuracy_rms_board_mean_mm"])), "",
                  "| # | Error source | Class | Input (1 sigma) | At tip (mm RMS) | Share | Source of the input |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for i, term in enumerate(r["terms"], 1):
            if term["tip_rms_mm"] < 0.0005:
                continue
            shown = term["input_unit"] if term["input_unit"] == "fit covariance" else "%s %s" % (_f(term["input_sigma"], 4), term["input_unit"])
            lines.append("| %d | %s | %s | %s | %s | %.0f%% | %s |" % (
                i, term["label"], term["class"], shown,
                _f(term["tip_rms_mm"]), 100 * term["share"], term["source"]))
        top = r["terms"][:3]
        lines += ["", "**Fix first:**", ""]
        for term in top:
            lines.append("- **%s** (%.0f%%). Measure: %s Fix: %s" % (term["label"], 100 * term["share"], term["measure"], term["fix"]))
        lines.append("")

    sw = rep["distance_sweep"]
    if sw["rows"]:
        lines += ["## Board distance from the base", "",
                  "Scenario: %s. The board is moved along its current direction from the base. "
                  "J1's error at the nozzle grows with distance from the base; J2's and the cameras' don't." % sw["scenario"], "",
                  "| Board centre radius (mm) | Accuracy RMS (mm) |", "| --- | --- |"]
        lines += ["| %d | %s |" % (row["board_centre_radius_mm"], _f(row["accuracy_rms_mm"])) for row in sw["rows"]]
        lines.append("")

    lines += ["## Assumptions to replace with measurements", "",
              "Edit these in the config once measured, and change each source to say how.", "",
              "| Input | Value | Note |", "| --- | --- | --- |"]
    lines += ["| `%s` | %s | %s |" % (path, json.dumps(value), note) for path, value, note in rep["assumptions"]]
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", action="append", default=[], metavar="dotted.key=value",
                    help="override an input, e.g. joints.j1.microsteps=32 (JSON values)")
    ap.add_argument("--arm-calibration", help="arm_calibration.json from kinematic_fit.py: use measured geometry and fit errors")
    ap.add_argument("--out", help="write the markdown report here instead of stdout")
    ap.add_argument("--json", help="also write the full result as JSON")
    a = ap.parse_args(argv)
    overrides = []
    for item in a.set:
        key, _, val = item.partition("=")
        overrides.append((key.strip(), _parse_cli_value(val.strip())))
    rep = run(a.config, overrides, a.arm_calibration)
    md = to_markdown(rep)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8", newline="\n")
    else:
        print(md)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
