"""
reducer_inspect.py — incoming inspection of a strain-wave reducer before the
second one is bought (ADR-001, action items 1-2).

Two bench tests, both with the reducer's output facing a camera:

1. Transmission error (TE): step the input monotonically (always the same
   direction, so lost motion doesn't enter) and measure the real output
   angle each time. TE is what is left after removing a constant offset
   and a straight-line ratio error.
     coarse sweep: one full output turn (catches once-per-output-turn error)
     fine sweep:   two input turns (catches the strain wave's own ripple)
   The output angle comes from a ChArUco target glued flat to the output
   flange, centred on the axis, photographed after each step with the
   calibrated camera (charuco_calib.py intrinsics). Or type angles in from
   any other reference.
2. Lost motion and stiffness: lock the input (motor energised at full
   current), hang weights on a lever on the output in both directions,
   read the angle at each load (camera, or a dial indicator at a known
   radius).

Then the gate: the measured TE and lost motion go into budget_config.json
(the reducer option in the scenarios), the error budget reruns, and the
report says whether the arm still meets its targets with this reducer.

    python -m metrology.reducer_inspect plan --out te_plan.csv
    python -m metrology.reducer_inspect angles te_plan.csv --images shots/ --intrinsics top_camera.json --square-mm 10.02
    python -m metrology.reducer_inspect analyze te_plan.csv --lost-motion lm.csv --out reducer_report.md --update-config
    python -m metrology.reducer_inspect simulate --out te_synthetic.csv       # pipeline check, marked synthetic
"""

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ARCMIN = 1.0 / 60.0                     # degrees
GATE_TE_ARCMIN = 2.0                    # ADR-001: half peak-to-peak TE must be <= 2 arcmin
NOISE_LIMIT_ARCMIN = 1.0                # per-position noise above this: the gate can't be judged
TE_FIELDS = ["seq", "sweep", "input_microsteps", "output_cmd_deg", "image", "angle_deg", "synthetic"]
LM_FIELDS = ["seq", "cycle", "torque_nm", "angle_deg", "dial_mm", "synthetic"]


# ─────────────────────────────────────────────────────────────────────────
# Plans
# ─────────────────────────────────────────────────────────────────────────

def plan_te(ratio=50, full_steps=200, microsteps=16, coarse_points=80, fine_input_revs=2, fine_points=64, frames=3):
    """Monotonic step positions for the coarse and fine sweeps, `frames`
    photos per position (averaged later: one photo reads the angle to about
    1 arcmin, which is too close to the 2 arcmin gate). Each sweep starts
    with an unmeasured lead-in move so its first point is approached in the
    same direction as the rest."""
    per_input_rev = full_steps * microsteps
    per_output_rev = int(round(per_input_rev * ratio))
    rows = []

    def add(sweep, steps, measured=True):
        for f in range(frames if measured else 1):
            rows.append({"seq": len(rows), "sweep": sweep if measured else sweep + "-lead-in",
                         "input_microsteps": int(steps), "output_cmd_deg": "%.6f" % (steps / per_output_rev * 360.0),
                         "image": "%s_%06d_f%d.png" % (sweep, steps, f) if measured else "", "angle_deg": "", "synthetic": ""})

    coarse_step = per_output_rev // coarse_points
    add("coarse", -per_input_rev, measured=False)
    for i in range(coarse_points):
        add("coarse", i * coarse_step)
    fine_step = fine_input_revs * per_input_rev // fine_points
    base = coarse_points * coarse_step + per_input_rev     # carry on in the same direction
    add("fine", base - per_input_rev // 4, measured=False)
    for i in range(fine_points):
        add("fine", base + i * fine_step)
    return rows


def plan_lost_motion(torque_nm=0.1, cycles=3):
    """0, +T, +2T, +T, 0, -T, -2T, -T, 0 per cycle. T about 3% of rated torque
    is the usual lost-motion load; 0.1 N*m suits a ~3.5 N*m rated unit."""
    levels = [0, 1, 2, 1, 0, -1, -2, -1, 0]
    return [{"seq": i, "cycle": c, "torque_nm": "%.4f" % (lv * torque_nm), "angle_deg": "", "dial_mm": "", "synthetic": ""}
            for i, (c, lv) in enumerate((c, lv) for c in range(1, cycles + 1) for lv in levels)]


# ─────────────────────────────────────────────────────────────────────────
# Output angle from images
# ─────────────────────────────────────────────────────────────────────────

def board_rotations(board, intrinsics, images):
    """Camera-frame rotation of the target in each image (None if not found)."""
    import cv2
    from metrology.charuco_calib import detect
    K = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    dist = np.asarray(intrinsics["dist_coeffs"], dtype=np.float64)
    out = []
    for im in images:
        det = None if im is None else detect(board, im)
        if det is None:
            out.append(None)
            continue
        obj, img, _ = det
        ok, rvec, _ = cv2.solvePnP(obj, img, K, dist)
        out.append(cv2.Rodrigues(rvec)[0] if ok else None)
    return out


def signed_angles(rotations):
    """Rotation of each pose about the common output axis, relative to the
    first pose, in degrees and unwrapped (so a full turn reads 0..360).

    The axis matters: a few degrees of axis error turns into arcminutes of
    angle error (it enters squared, times sin 2*theta). So it's averaged
    over every pose that turned 15-165 degrees, where each pose's axis is
    well determined, and falls back to the mean target normal (the target
    is glued flat on the flange) when the sweep is too short for that."""
    import cv2
    valid = [R for R in rotations if R is not None]
    if len(valid) < 2:
        raise ValueError("need at least two images where the target was found")
    R0 = valid[0]
    normal = np.mean([R[:, 2] for R in valid], axis=0)
    normal /= np.linalg.norm(normal)
    axes = []
    for R in valid[1:]:
        rv = cv2.Rodrigues(R @ R0.T)[0].ravel()
        ang = np.linalg.norm(rv)
        if math.radians(15) < ang < math.radians(165):
            u = rv / ang
            axes.append(u if u @ normal > 0 else -u)
    axis = np.mean(axes, axis=0) if len(axes) >= 2 else normal
    axis /= np.linalg.norm(axis)
    perp = lambda v: v - axis * (axis @ v)
    u0 = perp(R0[:, 0])
    out, prev = [], None
    for R in rotations:
        if R is None:
            out.append(None)
            continue
        v = perp(R[:, 0])
        a = math.degrees(math.atan2(axis @ np.cross(u0, v), u0 @ v))
        if prev is not None:
            a += 360.0 * round((prev - a) / 360.0)
        out.append(a)
        prev = a
    return out


# ─────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────

def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _filled(rows, key="angle_deg"):
    return [r for r in rows if (r.get(key) or "").strip()]


def _fourier(t, harmonics):
    cols = []
    for k in harmonics:
        cols += [np.cos(k * t), np.sin(k * t)]
    return np.column_stack(cols) if cols else np.zeros((len(t), 0))


def transmission_error(cmd_deg, meas_deg, phase_scale=1.0, harmonics=(1, 2, 3)):
    """Fit meas - cmd = offset + ratio error * cmd + Fourier series, with the
    series in (phase_scale * output angle): 1 for once-per-output-turn
    terms, the gear ratio for once-per-input-turn terms.

    The TE is the fitted periodic part, judged by its peak-to-peak on a
    dense grid, not by the raw readings: raw peak-to-peak grows with
    measurement noise and would fail a good reducer. What the fit doesn't
    explain is reported as measurement noise. Handles a target that reads
    the rotation backwards."""
    cmd = np.asarray(cmd_deg, dtype=float)
    meas = np.asarray(meas_deg, dtype=float)
    if np.corrcoef(cmd, meas)[0, 1] < 0:
        meas = -meas
    t = np.radians(cmd) * phase_scale
    A = np.column_stack([np.ones_like(cmd), cmd, _fourier(t, harmonics)])
    coef, *_ = np.linalg.lstsq(A, meas - cmd, rcond=None)
    resid = meas - cmd - A @ coef
    dof = max(1, len(cmd) - A.shape[1])
    sigma2 = resid @ resid / dof
    dense_basis = _fourier(np.linspace(0, 2 * np.pi, 721), harmonics)
    curve = dense_basis @ coef[2:]
    raw = meas - cmd - A[:, :2] @ coef[:2]
    # 95% interval on the peak-to-peak: redraw the coefficients from their
    # covariance. Noise pushes a fitted peak-to-peak up, so the upper bound
    # is conservative; the gate passes only on the upper bound.
    cov = sigma2 * np.linalg.inv(A.T @ A)[2:, 2:]
    draws = np.random.default_rng(0).multivariate_normal(coef[2:], cov, 2000, method="eigh")
    pps = np.ptp(draws @ dense_basis.T, axis=1) / ARCMIN
    return {"ratio_error": float(coef[1]), "harmonics": list(harmonics), "phase_scale": phase_scale,
            "coefficients_deg": [float(c) for c in coef[2:]],
            "amplitudes_arcmin": [float(np.hypot(coef[2 + 2 * i], coef[3 + 2 * i]) / ARCMIN) for i in range(len(harmonics))],
            "pp_arcmin": float(np.ptp(curve) / ARCMIN),
            "pp_ci95_arcmin": [float(np.percentile(pps, 2.5)), float(np.percentile(pps, 97.5))],
            "noise_arcmin": float(np.sqrt(sigma2) / ARCMIN),
            "raw_pp_arcmin": float(np.ptp(raw) / ARCMIN)}


def _positions(rows):
    """Average repeated photos of the same position. -> (cmd, angle, n_frames)."""
    groups = {}
    for r in rows:
        groups.setdefault(int(r["input_microsteps"]), (float(r["output_cmd_deg"]), []))[1].append(float(r["angle_deg"]))
    keys = sorted(groups)
    return (np.array([groups[k][0] for k in keys]), np.array([np.mean(groups[k][1]) for k in keys]),
            np.array([len(groups[k][1]) for k in keys]))


def lost_motion(rows, radius_mm=None):
    """Hysteresis at zero load (lost motion) and stiffness from the loaded
    points: angle = a + T / k + h * sign(T)."""
    T, ang = [], []
    for r in _filled(rows, "angle_deg") or _filled(rows, "dial_mm"):
        T.append(float(r["torque_nm"]))
        if (r.get("angle_deg") or "").strip():
            ang.append(float(r["angle_deg"]))
        else:
            if not radius_mm:
                raise ValueError("dial_mm readings need --radius-mm (dial distance from the output axis)")
            ang.append(math.degrees(float(r["dial_mm"]) / radius_mm))
    T, ang = np.array(T), np.array(ang)
    if len(T) < 6:
        raise ValueError("need at least 6 lost-motion readings")
    # sign of the last non-zero load before each zero reading
    last, after = 0.0, []
    for t in T:
        after.append(last)
        if t != 0:
            last = np.sign(t)
    after = np.array(after)
    zero = T == 0
    up, down = ang[zero & (after > 0)], ang[zero & (after < 0)]
    if not len(up) or not len(down):
        raise ValueError("need zero-load readings after both positive and negative loads")
    loaded = T != 0
    A = np.column_stack([np.ones(loaded.sum()), T[loaded], np.sign(T[loaded])])
    (a, inv_k, h), *_ = np.linalg.lstsq(A, ang[loaded], rcond=None)
    k = math.degrees(1.0) / inv_k if inv_k > 0 else float("inf")      # inv_k is degrees per N*m
    return {"lost_motion_deg": float(abs(up.mean() - down.mean())),
            "lost_motion_arcmin": float(abs(up.mean() - down.mean()) / ARCMIN),
            "stiffness_nm_per_rad": float(k), "loaded_hysteresis_arcmin": float(2 * abs(h) / ARCMIN),
            "n": int(len(T))}


def analyze(te_rows, lm_rows=None, radius_mm=None, ratio=50.0):
    filled = [r for r in _filled(te_rows) if r["sweep"] in ("coarse", "fine")]
    synthetic = any((r.get("synthetic") or "").strip() == "1" for r in filled + list(lm_rows or []))
    out = {"synthetic": synthetic, "ratio": ratio}
    for sweep, scale, harmonics in (("coarse", 1.0, (1, 2, 3)), ("fine", ratio, (1, 2))):
        rs = [r for r in filled if r["sweep"] == sweep]
        if not rs:
            continue
        cmd, ang, frames = _positions(rs)
        if len(cmd) < 2 * len(harmonics) + 4:
            raise ValueError("the %s sweep needs at least %d measured positions" % (sweep, 2 * len(harmonics) + 4))
        res = transmission_error(cmd - cmd[0], ang - ang[0], scale, harmonics)
        res["positions"], res["frames_per_position"] = int(len(cmd)), float(frames.mean())
        out[sweep] = res
    if "coarse" not in out:
        raise ValueError("no measured coarse sweep")
    c = out["coarse"]
    f = out.get("fine", {"pp_arcmin": 0.0, "pp_ci95_arcmin": [0.0, 0.0]})
    # The two parts can line up, so the bound adds them. A TE map in output
    # angle (the coarse fit) removes the first; the strain-wave ripple stays.
    out["te_half_pp_arcmin"] = (c["pp_arcmin"] + f["pp_arcmin"]) / 2.0
    out["te_half_pp_ci95_arcmin"] = [(c["pp_ci95_arcmin"][i] + f["pp_ci95_arcmin"][i]) / 2.0 for i in (0, 1)]
    out["te_half_pp_mapped_arcmin"] = f["pp_arcmin"] / 2.0
    out["te_half_pp_mapped_ci95_arcmin"] = [v / 2.0 for v in f["pp_ci95_arcmin"]]
    out["te_map"] = {"form": "te(theta) = sum over k of a_k cos(k theta) + b_k sin(k theta); theta = output angle; "
                             "coefficients a_1, b_1, a_2, b_2, ... in degrees",
                     "harmonics": c["harmonics"], "coefficients_deg": c["coefficients_deg"]}
    noise = max(out[s]["noise_arcmin"] for s in ("coarse", "fine") if s in out)
    out["noise_arcmin"] = noise
    out["noise_limited"] = noise > NOISE_LIMIT_ARCMIN
    if lm_rows:
        out["lost_motion"] = lost_motion(lm_rows, radius_mm)

    def verdict(ci):
        if out["noise_limited"]:
            return "noisy"
        if ci[1] <= GATE_TE_ARCMIN:
            return "pass"
        return "fail" if ci[0] > GATE_TE_ARCMIN else "uncertain"

    out["te_verdict"] = verdict(out["te_half_pp_ci95_arcmin"])
    out["te_verdict_with_map"] = verdict(out["te_half_pp_mapped_ci95_arcmin"])
    return out


# ─────────────────────────────────────────────────────────────────────────
# Gate: put the measurements into the budget and rerun it
# ─────────────────────────────────────────────────────────────────────────

def apply_to_budget(res, config_path, label, joints=("j1", "j2"), use_map=False, reducer_match="strain-wave"):
    """Write measured TE / lost motion / stiffness into the scenarios that
    fit this kind of reducer to these joints (not the belt option), rerun
    the budget, and return the recommended scenario's result. Both joints
    by default: the second unit is assumed to match the one measured."""
    from metrology import error_budget as eb
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    te_arcmin = res["te_half_pp_mapped_arcmin"] if use_map else res["te_half_pp_arcmin"]
    values = {"reducer_transmission_error_deg": (round(te_arcmin * ARCMIN, 5),
                                                 "measured: half peak-to-peak TE%s, %s" % (" after TE map" if use_map else "", label))}
    lm = res.get("lost_motion")
    if lm:
        values["reducer_backlash_deg"] = (round(lm["lost_motion_deg"], 5), "measured: lost motion at zero load, %s" % label)
        if math.isfinite(lm["stiffness_nm_per_rad"]):
            values["reducer_stiffness_nm_per_rad"] = (round(lm["stiffness_nm_per_rad"], 1), "measured: loaded stiffness, %s" % label)
    touched = 0
    for sc in raw.get("scenarios", []):
        for j in joints:
            reducer = sc.get("set", {}).get("joints.%s.reducer" % j, {})
            if reducer_match not in str(reducer.get("value", "") if isinstance(reducer, dict) else reducer):
                continue
            for key, (v, src) in values.items():
                dotted = "joints.%s.%s" % (j, key)
                if dotted in sc.get("set", {}):
                    sc["set"][dotted] = {"value": v, "source": src}
                    touched += 1
    if not touched:
        raise ValueError("no scenario in %s sets reducer values for %s" % (config_path, ", ".join(joints)))
    Path(config_path).write_text(eb.dump_config(raw) + "\n", encoding="utf-8")
    rep = eb.run(config_path)
    best = next((s for s in rep["scenarios"] if s["recommended"]), rep["scenarios"][-1])
    return {"scenario": best["name"], "accuracy_rms_mm": best["accuracy_rms_mm"],
            "repeatability_std_mm": best["repeatability_std_mm"],
            "meets_targets": best["meets_accuracy_target"] and best["meets_repeatability_target"],
            "target": rep["target"], "values_written": {k: v[0] for k, v in values.items()}}


def to_markdown(res, label, gate=None):
    lines = ["# Strain-wave reducer inspection: %s" % label, ""]
    if res["synthetic"]:
        lines += ["**SYNTHETIC DATA.** Simulated readings for checking the pipeline. They describe no real reducer.", ""]
    lines += ["## Transmission error", "",
              "| Sweep | Positions x photos | Fitted TE peak-to-peak | Measurement noise | Raw peak-to-peak | Ratio error |",
              "| --- | --- | --- | --- | --- | --- |"]
    names = {"coarse": "Coarse (one output turn; 1-3x per output turn)", "fine": "Fine (two input turns; 1-2x per input turn)"}
    for sweep in ("coarse", "fine"):
        if sweep in res:
            r = res[sweep]
            lines.append("| %s | %d x %.0f | %.2f arcmin | %.2f arcmin | %.2f arcmin | %+.2e |" % (
                names[sweep], r["positions"], r["frames_per_position"], r["pp_arcmin"], r["noise_arcmin"],
                r["raw_pp_arcmin"], r["ratio_error"]))
    ci, cim = res["te_half_pp_ci95_arcmin"], res["te_half_pp_mapped_ci95_arcmin"]
    lines += ["", "TE for the budget (half of coarse + fine peak-to-peak): **%.2f arcmin**, 95%% interval %.2f-%.2f "
              "(gate: <= %.1f). With the coarse fit applied as a TE map: %.2f arcmin, %.2f-%.2f." % (
                  res["te_half_pp_arcmin"], ci[0], ci[1], GATE_TE_ARCMIN, res["te_half_pp_mapped_arcmin"], cim[0], cim[1]),
              "", "Raw peak-to-peak includes measurement noise and is shown for reference only. A measurement error "
              "that repeats with the same period as the TE can't be told apart from it, so it counts as TE.", ""]
    if res["noise_limited"]:
        lines += ["**Measurement too noisy to judge** (%.2f arcmin per position, limit %.1f): take more photos per "
                  "position, use a larger target or move the camera closer, then measure again." % (
                      res["noise_arcmin"], NOISE_LIMIT_ARCMIN), ""]
    if "lost_motion" in res:
        lm = res["lost_motion"]
        lines += ["## Lost motion and stiffness", "",
                  "Lost motion at zero load: **%.2f arcmin** (%d readings). Loaded stiffness: %.0f N*m/rad. "
                  "Hysteresis between loaded branches: %.2f arcmin." % (
                      lm["lost_motion_arcmin"], lm["n"], lm["stiffness_nm_per_rad"], lm["loaded_hysteresis_arcmin"]), ""]
    words = {"pass": "PASS", "fail": "FAIL", "noisy": "CAN'T JUDGE: measurement too noisy",
             "uncertain": "UNCERTAIN: the 95% interval straddles the gate; measure more positions or photos"}
    verdict = words[res["te_verdict"]]
    if res["te_verdict"] in ("fail", "uncertain") and res["te_verdict_with_map"] == "pass":
        verdict += " (PASS with the TE map applied)"
    lines += ["## Gate (ADR-001)", "", "TE: **%s**." % verdict]
    if gate:
        lines.append("Error budget rerun with these values: %s predicts **%.3f mm RMS**, %.3f mm std "
                     "(targets %s / %s mm): **%s**." % (
                         gate["scenario"], gate["accuracy_rms_mm"], gate["repeatability_std_mm"],
                         gate["target"]["accuracy_rms_mm"], gate["target"]["repeatability_std_mm"],
                         "meets targets" if gate["meets_targets"] else "does not meet targets"))
    lines += ["", "Buy the second reducer only if both are a pass. Otherwise see ADR-001, \"What would change this\".", ""]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────────────────

def simulate_te(rows, ratio=50, te_out_arcmin=1.5, te_in_arcmin=0.4, noise_arcmin=0.1, seed=0):
    """Fill a TE plan with SYNTHETIC angles: once-per-output-turn error,
    twice-per-input-turn ripple and measurement noise."""
    rng = np.random.default_rng(seed)
    out = []
    for r in rows:
        r = dict(r)
        if r["sweep"] in ("coarse", "fine"):
            th = math.radians(float(r["output_cmd_deg"]))
            te = te_out_arcmin * math.sin(th + 0.4) + te_in_arcmin * math.sin(2 * ratio * th + 1.1)   # amplitudes
            r["angle_deg"] = "%.6f" % (float(r["output_cmd_deg"]) + (te + rng.normal(0, noise_arcmin)) * ARCMIN)
            r["synthetic"] = "1"
        out.append(r)
    return out


def simulate_lost_motion(rows, lost_motion_arcmin=1.5, stiffness=2200.0, noise_arcmin=0.05, seed=0):
    rng = np.random.default_rng(seed)
    out, last = [], 1.0
    for r in rows:
        r = dict(r)
        T = float(r["torque_nm"])
        if T:
            last = math.copysign(1.0, T)
        a = math.degrees(T / stiffness) + last * lost_motion_arcmin / 2 * ARCMIN + rng.normal(0, noise_arcmin) * ARCMIN
        r["angle_deg"], r["synthetic"] = "%.6f" % a, "1"
        out.append(r)
    return out


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="Strain-wave reducer incoming inspection (ADR-001).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--lost-motion-out", help="also write a lost-motion load sequence")
    p.add_argument("--ratio", type=float, default=50)
    p.add_argument("--full-steps", type=int, default=200)
    p.add_argument("--microsteps", type=int, default=16)
    p.add_argument("--torque-nm", type=float, default=0.1)
    p.add_argument("--frames", type=int, default=3, help="photos per position (averaged)")

    g = sub.add_parser("angles", help="fill angle_deg from the images named in the plan")
    g.add_argument("plan_csv")
    g.add_argument("--images", required=True)
    g.add_argument("--intrinsics", required=True)
    g.add_argument("--squares-x", type=int, default=9)
    g.add_argument("--squares-y", type=int, default=6)
    g.add_argument("--square-mm", type=float, default=10.0)
    g.add_argument("--marker-mm", type=float, default=7.5)
    g.add_argument("--dictionary", default="DICT_5X5_100")

    z = sub.add_parser("analyze")
    z.add_argument("plan_csv")
    z.add_argument("--lost-motion", help="filled lost-motion CSV")
    z.add_argument("--radius-mm", type=float, help="dial indicator distance from the axis, for dial_mm readings")
    z.add_argument("--ratio", type=float, default=50)
    z.add_argument("--label", default="", help="e.g. supplier, model and serial number")
    z.add_argument("--update-config", nargs="?", const=os.path.join(os.path.dirname(__file__), "budget_config.json"),
                   help="write the measured values into the budget config (default: metrology/budget_config.json) and rerun it")
    z.add_argument("--use-map", action="store_true", help="budget with the TE left after the TE map")
    z.add_argument("--out", help="markdown report (default: stdout)")
    z.add_argument("--json")

    s = sub.add_parser("simulate", help="SYNTHETIC filled plans, for checking the pipeline")
    s.add_argument("--out", required=True)
    s.add_argument("--lost-motion-out")
    s.add_argument("--te-arcmin", type=float, default=1.5)
    a = ap.parse_args(argv)

    if a.cmd == "plan":
        rows = plan_te(a.ratio, a.full_steps, a.microsteps, frames=a.frames)
        _write(a.out, rows, TE_FIELDS)
        print("Wrote %d moves to %s. Step the input to each input_microsteps value in order (always the same "
              "direction), let it settle, save a photo under the 'image' name." % (len(rows), a.out))
        if a.lost_motion_out:
            _write(a.lost_motion_out, plan_lost_motion(a.torque_nm), LM_FIELDS)
            print("Wrote the lost-motion sequence to %s. Lever mass for %.3f N*m at 50 mm: %.0f g."
                  % (a.lost_motion_out, a.torque_nm, a.torque_nm / (9.81 * 0.05) * 1000))
        return 0

    if a.cmd == "simulate":
        _write(a.out, simulate_te(plan_te(), te_out_arcmin=a.te_arcmin), TE_FIELDS)
        if a.lost_motion_out:
            _write(a.lost_motion_out, simulate_lost_motion(plan_lost_motion()), LM_FIELDS)
        print("Wrote SYNTHETIC readings to %s%s." % (a.out, " and " + a.lost_motion_out if a.lost_motion_out else ""))
        return 0

    if a.cmd == "angles":
        import cv2
        from metrology.charuco_calib import make_board
        rows = _read(a.plan_csv)
        board = make_board((a.squares_x, a.squares_y), a.square_mm, a.marker_mm, a.dictionary)
        intr = json.loads(Path(a.intrinsics).read_text(encoding="utf-8"))
        for sweep in ("coarse", "fine"):
            rs = [r for r in rows if r["sweep"] == sweep and r["image"]]
            imgs = [cv2.imread(os.path.join(a.images, r["image"]), cv2.IMREAD_GRAYSCALE) for r in rs]
            for r, ang in zip(rs, signed_angles(board_rotations(board, intr, imgs))):
                r["angle_deg"] = "" if ang is None else "%.6f" % ang
        _write(a.plan_csv, rows, TE_FIELDS)
        missing = sum(1 for r in rows if r["image"] and not r["angle_deg"])
        print("Filled angles in %s%s." % (a.plan_csv, " (%d images without a readable target)" % missing if missing else ""))
        return 0

    res = analyze(_read(a.plan_csv), _read(a.lost_motion) if a.lost_motion else None, a.radius_mm, a.ratio)
    label = (a.label or "unlabelled unit") + ", " + dt.date.today().isoformat()
    if res["synthetic"]:
        label = "SYNTHETIC " + label
    gate = None
    if a.update_config:
        if res["synthetic"] and Path(a.update_config).resolve() == (Path(__file__).parent / "budget_config.json").resolve():
            raise SystemExit("refusing to write SYNTHETIC readings into the real budget_config.json; pass a copy's path")
        gate = apply_to_budget(res, a.update_config, label, use_map=a.use_map)
    md = to_markdown(res, label, gate)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8", newline="\n")
        print("Wrote %s" % a.out)
    else:
        print(md)
    if a.json:
        Path(a.json).write_text(json.dumps({**res, "gate": gate}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
