"""
kinematic_fit.py — measure the arm's real geometry and replace the
ARM_BASE_OFFSET_MM placeholder.

Procedure (needs the assembled arm and a calibrated top camera):

1. Plan poses, spread over the board area plus a margin:
       python -m metrology.kinematic_fit plan --out cal_poses.csv
2. For each row, command the joints to q1_cmd_deg / q2_cmd_deg, ALWAYS
   approaching from the same side (e.g. from -5 deg on both joints), wait
   for the 150 ms settle, then measure where the nozzle tip really is in the
   BOARD frame (mm) and fill x_meas_mm / y_meas_mm. Ways to measure, best first:
     - top camera (after charuco_calib.py) on a small target on the nozzle tip
     - nozzle pricks a dot on paper taped to the board, scanned at 1200 dpi
       together with a ruler, dots located in the scan
   Use board coordinates with y pointing up (right-handed). Raw image pixel
   coordinates have y down; the fit detects that and warns.
3. Fit:
       python -m metrology.kinematic_fit fit cal_poses.csv --out arm_calibration.json
   Copy arm_calibration.json to the backend root (or set ARM_CALIBRATION_PATH):
   layout_engine then uses the measured link lengths and base offset.
   Feed it to the error budget too:
       python -m metrology.error_budget --arm-calibration arm_calibration.json

Without hardware, `simulate` fills the CSV from a made-up arm so the
pipeline can be exercised. Its output is marked synthetic, and layout_engine
refuses to load a synthetic calibration.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from metrology.scara import ArmGeometry, fit_geometry, forward, plan_calibration_poses

FIELDS = ["pose", "q1_cmd_deg", "q2_cmd_deg", "x_nominal_mm", "y_nominal_mm", "x_meas_mm", "y_meas_mm", "synthetic"]


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    usable = [r for r in rows if (r.get("x_meas_mm") or "").strip() and (r.get("y_meas_mm") or "").strip()]
    q = np.array([[float(r["q1_cmd_deg"]), float(r["q2_cmd_deg"])] for r in usable])
    xy = np.array([[float(r["x_meas_mm"]), float(r["y_meas_mm"])] for r in usable])
    synthetic = any((r.get("synthetic") or "").strip() in ("1", "true", "yes") for r in usable)
    return q, xy, synthetic, len(rows) - len(usable)


def write_rows(path, xy_nom, q, xy_meas=None, synthetic=False):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, ((x, y), (a, b)) in enumerate(zip(xy_nom, q)):
            row = {"pose": i, "q1_cmd_deg": "%.4f" % a, "q2_cmd_deg": "%.4f" % b,
                   "x_nominal_mm": "%.3f" % x, "y_nominal_mm": "%.3f" % y,
                   "x_meas_mm": "", "y_meas_mm": "", "synthetic": "1" if synthetic else ""}
            if xy_meas is not None:
                row["x_meas_mm"], row["y_meas_mm"] = "%.4f" % xy_meas[i][0], "%.4f" % xy_meas[i][1]
            w.writerow(row)


def calibration_record(result, csv_path, synthetic, nominal):
    g = result.geometry
    return {
        "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "synthetic": synthetic,
        "source_csv": Path(csv_path).name,
        "source_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        "model": "planar 2R; p_arm = p_board + base_offset; theta_i = scale_i * q_i + zero_i (see metrology/scara.py)",
        "geometry": g.to_dict(),
        "nominal_geometry": nominal.to_dict(),
        "fit": result.summary(),
    }


def report(record, skipped=0):
    f, g = record["fit"], record["geometry"]
    lines = []
    if record["synthetic"]:
        lines.append("SYNTHETIC DATA: this calibration describes a made-up arm, not yours.")
    lines.append("Poses used: %d%s" % (f["n_points"], " (%d rows without measurements skipped)" % skipped if skipped else ""))
    lines.append("Base offset (board origin in arm frame): (%.3f, %.3f) mm  +/- (%.3f, %.3f)"
                 % (g["base_offset_mm"][0], g["base_offset_mm"][1], f["std_error"]["base_x_mm"], f["std_error"]["base_y_mm"]))
    lines.append("Link lengths: L1 %.3f +/- %.3f mm, L2 %.3f +/- %.3f mm"
                 % (g["l1_mm"], f["std_error"]["l1_mm"], g["l2_mm"], f["std_error"]["l2_mm"]))
    lines.append("Joint zeros: %.3f +/- %.3f deg, %.3f +/- %.3f deg"
                 % (g["zero_deg"][0], f["std_error"]["zero1_deg"], g["zero_deg"][1], f["std_error"]["zero2_deg"]))
    if "scale1" in f["std_error"]:
        lines.append("Joint scale: %.5f +/- %.5f, %.5f +/- %.5f"
                     % (g["scale"][0], f["std_error"]["scale1"], g["scale"][1], f["std_error"]["scale2"]))
    else:
        lines.append("Joint scale: fixed at 1 (needs 16+ poses and 30+ deg of travel per joint to fit)")
    lines.append("Residual RMS %.4f mm, max %.4f mm; held-out RMS %s mm"
                 % (f["residual_rms_mm"], f["residual_max_mm"], f["heldout_rms_mm"] if f["heldout_rms_mm"] is not None else "n/a"))
    if f["max_abs_correlation"] > 0.95:
        lines.append("Note: two parameters are %.0f%% correlated. Each is uncertain alone, but their combined effect on the "
                     "board is not; spread the poses wider to separate them." % (100 * f["max_abs_correlation"]))
    reversed_joints = [i + 1 for i, s in enumerate(g["scale"]) if s < 0]
    if len(reversed_joints) == 1:
        lines.append("WARNING: J%d moves opposite to its command (scale < 0). Reverse that motor's direction "
                     "in firmware or wiring, then measure again." % reversed_joints[0])
    if f["mirrored_fits_better"]:
        lines.append("WARNING: the data fits far better with the board's y axis flipped. The measurements are probably "
                     "in image coordinates (y down) or a mirrored frame. Convert to y-up board mm and fit again.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plan, simulate and fit the arm's kinematic calibration.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="write poses to command and measure")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--margin-mm", type=float, default=15.0)
    p.add_argument("--board-mm", type=float, nargs=2, default=(62.0, 42.0))
    p.add_argument("--elbow", type=int, choices=(-1, 1), default=1)

    s = sub.add_parser("simulate", help="fill a plan with SYNTHETIC measurements from a made-up arm")
    s.add_argument("plan_csv")
    s.add_argument("--out", required=True)
    s.add_argument("--noise-mm", type=float, default=0.02)
    s.add_argument("--seed", type=int, default=1)

    f = sub.add_parser("fit", help="fit geometry to a measured CSV")
    f.add_argument("csv")
    f.add_argument("--out", default="arm_calibration.json")
    f.add_argument("--fit-scale", choices=("auto", "yes", "no"), default="auto")

    a = ap.parse_args(argv)
    nominal = ArmGeometry()

    if a.cmd == "plan":
        xy, q = plan_calibration_poses(nominal, tuple(a.board_mm), n=a.n, margin_mm=a.margin_mm, elbow=a.elbow)
        write_rows(a.out, xy, q)
        print("Wrote %d poses to %s. Command each, measure the tip in board mm, fill x_meas_mm / y_meas_mm." % (len(q), a.out))
        return 0

    if a.cmd == "simulate":
        with open(a.plan_csv, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        q = np.array([[float(r["q1_cmd_deg"]), float(r["q2_cmd_deg"])] for r in rows])
        xy_nom = np.array([[float(r["x_nominal_mm"]), float(r["y_nominal_mm"])] for r in rows])
        made_up = ArmGeometry((186.0, 44.5), 151.2, 178.9, (-1.4, 0.8), (1.0, 1.0))
        meas = forward(made_up, q) + np.random.default_rng(a.seed).normal(0.0, a.noise_mm, q.shape)
        write_rows(a.out, xy_nom, q, meas, synthetic=True)
        print("Wrote SYNTHETIC measurements to %s (made-up arm %s, noise %.3f mm)." % (a.out, made_up.to_dict(), a.noise_mm))
        return 0

    q, xy, synthetic, skipped = read_rows(a.csv)
    fit_scale = None if a.fit_scale == "auto" else a.fit_scale == "yes"
    result = fit_geometry(q, xy, initial=nominal, fit_scale=fit_scale)
    record = calibration_record(result, a.csv, synthetic, nominal)
    Path(a.out).write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(report(record, skipped))
    print("Wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
