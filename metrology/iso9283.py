"""
iso9283.py — ISO 9283-style accuracy and repeatability test for the arm,
measured in the board plane.

What it computes, per pose (ISO 9283 definitions, restricted to x/y):
  AP   pose accuracy      distance from the commanded point to the MEAN of
                          the attained points
  RP   pose repeatability l_bar + 3*S_l, where l_j is the distance of each
                          attained point from that mean
  vAP  multi-directional  largest distance between the mean attained points
       accuracy variation when the same pose is approached from different
                          directions (exposes backlash and friction)
plus the two numbers the team's own plan uses (hardware_plan_v3, section 9):
RMS deviation from commanded, and per-axis standard deviation.

Poses: P1 at the centre of the test rectangle, P2-P5 on its diagonals at 10%
of the diagonal from the corners, as in the standard's measuring plane.
30 cycles of P1->P5 by default, as in the standard.

Deviations from ISO 9283, stated so the report can't be mistaken for a
certified test: planar (x, y) only, no orientation; the vAP directions are
in-plane (+/-x, +/-y) rather than the three base-axis directions; conditions
(speed, load, warm-up) are whatever you record with --conditions.

    python -m metrology.iso9283 plan --out iso_plan.csv [--arm-calibration arm_calibration.json]
    (run it on the arm, fill x_meas_mm / y_meas_mm)
    python -m metrology.iso9283 analyze iso_plan.csv --conditions "50% speed, no part, 30 min warm-up, 21 C" --out iso_report.md
    python -m metrology.iso9283 simulate iso_plan.csv --out iso_synthetic.csv     # pipeline check only
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from metrology.scara import ArmGeometry, forward, inverse

FIELDS = ["seq", "test", "cycle", "pose", "approach", "x_cmd_mm", "y_cmd_mm", "x_pre_mm", "y_pre_mm",
          "q1_cmd_deg", "q2_cmd_deg", "x_meas_mm", "y_meas_mm", "synthetic"]
DIRECTIONS = {"-x": (-1.0, 0.0), "+x": (1.0, 0.0), "-y": (0.0, -1.0), "+y": (0.0, 1.0)}


def iso_poses(x0, y0, x1, y1):
    """P1 centre; P2..P5 on the diagonals, 10% of the diagonal in from each corner."""
    c = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    corners = [np.array(p, dtype=float) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    poses = {"P1": c}
    for k, corner in enumerate(corners):
        opposite = corners[(k + 2) % 4]
        poses["P%d" % (k + 2)] = corner + 0.1 * (opposite - corner)
    return poses


def plan(rect, geometry, elbow=1, cycles=30, vap_cycles=10, vap_poses=("P1", "P2", "P4"),
         directions=("-x", "+x", "-y", "+y"), approach_mm=10.0):
    """Rows to run, in order. AP/RP rows go straight from the previous pose
    (the standard's cyclic P1->P5 path), after one unmeasured move to P5 so
    the first P1 is approached the same way as every later one. vAP rows
    first move to a pre-pose `approach_mm` away, then travel to the target
    in the stated direction."""
    poses = iso_poses(*rect)
    rows = []

    def add(test, cycle, name, approach, pre):
        target = poses[name]
        q = inverse(geometry, target[None], elbow=elbow)[0]
        if np.isnan(q).any():
            raise ValueError("%s %s is out of reach with this geometry" % (name, target.tolist()))
        rows.append({"seq": len(rows), "test": test, "cycle": cycle, "pose": name, "approach": approach,
                     "x_cmd_mm": "%.3f" % target[0], "y_cmd_mm": "%.3f" % target[1],
                     "x_pre_mm": "" if pre is None else "%.3f" % pre[0], "y_pre_mm": "" if pre is None else "%.3f" % pre[1],
                     "q1_cmd_deg": "%.4f" % q[0], "q2_cmd_deg": "%.4f" % q[1],
                     "x_meas_mm": "", "y_meas_mm": "", "synthetic": ""})

    add("lead-in", 0, "P5", "sequence", None)   # move only; not measured or analysed
    for cyc in range(1, cycles + 1):
        for name in ("P1", "P2", "P3", "P4", "P5"):
            add("AP_RP", cyc, name, "sequence", None)
    for name in vap_poses:
        for d in directions:
            pre = poses[name] - approach_mm * np.array(DIRECTIONS[d])   # final move travels in direction d
            for cyc in range(1, vap_cycles + 1):
                add("vAP", cyc, name, d, pre)
    return rows


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _measured(rows):
    return [r for r in rows if (r.get("x_meas_mm") or "").strip() and (r.get("y_meas_mm") or "").strip()]


def pose_stats(cmd, pts):
    """ISO 9283 AP and RP for one pose, plus per-axis std, from n attained points."""
    pts = np.asarray(pts, dtype=float)
    mean = pts.mean(axis=0)
    l = np.linalg.norm(pts - mean, axis=1)
    s_l = float(l.std(ddof=1)) if len(l) > 1 else 0.0
    return {"n": len(pts), "mean_mm": mean.tolist(), "AP_mm": float(np.linalg.norm(mean - cmd)),
            "APx_mm": float(mean[0] - cmd[0]), "APy_mm": float(mean[1] - cmd[1]),
            "l_bar_mm": float(l.mean()), "S_l_mm": s_l, "RP_mm": float(l.mean() + 3 * s_l),
            "std_x_mm": float(pts[:, 0].std(ddof=1)) if len(pts) > 1 else 0.0,
            "std_y_mm": float(pts[:, 1].std(ddof=1)) if len(pts) > 1 else 0.0}


def _bootstrap_ci(cmd, pts, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts, dtype=float)
    ap, rp = [], []
    for _ in range(n_boot):
        s = pts[rng.integers(0, len(pts), len(pts))]
        st = pose_stats(cmd, s)
        ap.append(st["AP_mm"])
        rp.append(st["RP_mm"])
    return [float(np.percentile(ap, 2.5)), float(np.percentile(ap, 97.5))], \
           [float(np.percentile(rp, 2.5)), float(np.percentile(rp, 97.5))]


def analyze(rows, accuracy_target_mm=0.15, repeat_target_mm=0.10):
    rows = [r for r in rows if r["test"] in ("AP_RP", "vAP")]
    meas = _measured(rows)
    if not meas:
        raise ValueError("no measured rows (x_meas_mm / y_meas_mm are empty)")
    synthetic = any((r.get("synthetic") or "").strip() in ("1", "true", "yes") for r in meas)
    xy = lambda r, a, b: np.array([float(r[a]), float(r[b])])

    out = {"synthetic": synthetic, "n_measurements": len(meas), "n_planned": len(rows), "poses": {}, "vap": {}}
    aprp = [r for r in meas if r["test"] == "AP_RP"]
    all_dev = []
    for name in sorted({r["pose"] for r in aprp}):
        rs = [r for r in aprp if r["pose"] == name]
        cmd = xy(rs[0], "x_cmd_mm", "y_cmd_mm")
        pts = np.array([xy(r, "x_meas_mm", "y_meas_mm") for r in rs])
        st = pose_stats(cmd, pts)
        if len(pts) >= 5:
            st["AP_ci95_mm"], st["RP_ci95_mm"] = _bootstrap_ci(cmd, pts)
        st["cmd_mm"] = cmd.tolist()
        out["poses"][name] = st
        all_dev.append(np.linalg.norm(pts - cmd, axis=1))

    for name in sorted({r["pose"] for r in meas if r["test"] == "vAP"}):
        rs = [r for r in meas if r["test"] == "vAP" and r["pose"] == name]
        cmd = xy(rs[0], "x_cmd_mm", "y_cmd_mm")
        means = {}
        for d in sorted({r["approach"] for r in rs}):
            pts = np.array([xy(r, "x_meas_mm", "y_meas_mm") for r in rs if r["approach"] == d])
            means[d] = pts.mean(axis=0)
        pairs = [(a, b) for i, a in enumerate(means) for b in list(means)[i + 1:]]
        vap = max((float(np.linalg.norm(means[a] - means[b])), a, b) for a, b in pairs) if pairs else (0.0, "", "")
        out["vap"][name] = {"vAP_mm": vap[0], "between": [vap[1], vap[2]],
                            "means_mm": {d: m.tolist() for d, m in means.items()}, "cmd_mm": cmd.tolist()}

    if all_dev:
        dev = np.concatenate(all_dev)
        out["rms_deviation_mm"] = float(np.sqrt(np.mean(dev ** 2)))
        out["max_std_axis_mm"] = max(max(p["std_x_mm"], p["std_y_mm"]) for p in out["poses"].values())
        out["max_AP_mm"] = max(p["AP_mm"] for p in out["poses"].values())
        out["max_RP_mm"] = max(p["RP_mm"] for p in out["poses"].values())
        out["meets_accuracy_target"] = out["rms_deviation_mm"] <= accuracy_target_mm
        out["meets_repeatability_target"] = out["max_std_axis_mm"] <= repeat_target_mm
    out["targets"] = {"accuracy_rms_mm": accuracy_target_mm, "repeatability_std_mm": repeat_target_mm,
                      "source": "hardware_plan_v3 section 9"}
    return out


def to_markdown(res, conditions=""):
    lines = ["# ISO 9283-style accuracy and repeatability", ""]
    if res["synthetic"]:
        lines += ["**SYNTHETIC DATA.** These measurements were simulated from a made-up arm to check the pipeline. "
                  "They say nothing about the real MiniMEE.", ""]
    lines += ["Conditions: %s" % (conditions or "**not recorded** (speed, load, warm-up and temperature belong here)"),
              "", "Measurements: %d of %d planned rows." % (res["n_measurements"], res["n_planned"]), ""]
    if "rms_deviation_mm" in res:
        t = res["targets"]
        lines += ["## Against the team's targets (%s)" % t["source"], "",
                  "| Metric | Measured | Target | Pass |", "| --- | --- | --- | --- |",
                  "| RMS deviation from commanded | %.4f mm | < %s mm | %s |" % (
                      res["rms_deviation_mm"], t["accuracy_rms_mm"], "yes" if res["meets_accuracy_target"] else "no"),
                  "| Largest per-axis std at one pose | %.4f mm | < %s mm | %s |" % (
                      res["max_std_axis_mm"], t["repeatability_std_mm"], "yes" if res["meets_repeatability_target"] else "no"),
                  "", "## Per pose (ISO 9283 AP and RP)", "",
                  "| Pose | Commanded (mm) | n | AP (mm) [95% CI] | APx, APy (mm) | RP (mm) [95% CI] | std x, y (mm) |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        ci = lambda v: "" if v is None else " [%.4f, %.4f]" % tuple(v)
        for name, p in res["poses"].items():
            lines.append("| %s | (%.2f, %.2f) | %d | %.4f%s | %+.4f, %+.4f | %.4f%s | %.4f, %.4f |" % (
                name, p["cmd_mm"][0], p["cmd_mm"][1], p["n"], p["AP_mm"], ci(p.get("AP_ci95_mm")),
                p["APx_mm"], p["APy_mm"], p["RP_mm"], ci(p.get("RP_ci95_mm")), p["std_x_mm"], p["std_y_mm"]))
        lines.append("")
    if res["vap"]:
        lines += ["## Multi-directional accuracy variation (vAP)", "",
                  "| Pose | vAP (mm) | Between approaches | Mean offset per approach (mm) |", "| --- | --- | --- | --- |"]
        for name, v in res["vap"].items():
            offs = ", ".join("%s (%+.3f, %+.3f)" % (d, m[0] - v["cmd_mm"][0], m[1] - v["cmd_mm"][1]) for d, m in v["means_mm"].items())
            lines.append("| %s | %.4f | %s | %s |" % (name, v["vAP_mm"], " vs ".join(v["between"]), offs))
        lines += ["", "A large vAP with a small RP means direction-dependent error (backlash, friction): approaching "
                  "every target from one side removes most of it.", ""]
    lines += ["Planar adaptation of ISO 9283: x/y only, in-plane vAP directions. CIs are bootstrap over cycles.", ""]
    return "\n".join(lines)


def simulate(rows, true_geom, nominal_geom, noise_mm=0.02, backlash_deg=(0.1, 0.1), seed=0):
    """Fill rows with SYNTHETIC attained positions. Commands were planned on
    `nominal_geom`; the 'real' arm is `true_geom`, plus per-joint backlash
    (the joint lags by half the play behind its last direction of travel)
    and per-axis Gaussian noise. For testing the analysis, nothing else."""
    rng = np.random.default_rng(seed)
    last_q = None
    out = []
    for r in rows:
        r = dict(r)
        q = np.array([float(r["q1_cmd_deg"]), float(r["q2_cmd_deg"])])
        if r["x_pre_mm"]:
            prev = inverse(nominal_geom, np.array([[float(r["x_pre_mm"]), float(r["y_pre_mm"])]]))[0]
        else:
            prev = last_q if last_q is not None else q - 1.0
        sign = np.sign(q - prev)
        sign[sign == 0] = 1.0
        q_real = q - sign * np.asarray(backlash_deg) / 2.0
        p = forward(true_geom, q_real[None])[0] + rng.normal(0.0, noise_mm, 2)
        r["x_meas_mm"], r["y_meas_mm"], r["synthetic"] = "%.4f" % p[0], "%.4f" % p[1], "1"
        last_q = q
        out.append(r)
    return out


def _geometry(path):
    if not path:
        return ArmGeometry(), False
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ArmGeometry.from_dict(data["geometry"]), True


def main(argv=None):
    ap = argparse.ArgumentParser(description="ISO 9283-style planar accuracy/repeatability test.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--rect", type=float, nargs=4, default=(0.0, 0.0, 62.0, 42.0), metavar=("X0", "Y0", "X1", "Y1"),
                   help="test rectangle in board mm (default: the 62x42 board)")
    p.add_argument("--cycles", type=int, default=30)
    p.add_argument("--vap-cycles", type=int, default=10)
    p.add_argument("--approach-mm", type=float, default=10.0)
    p.add_argument("--elbow", type=int, choices=(-1, 1), default=1)
    p.add_argument("--arm-calibration", help="plan joint commands on measured geometry (recommended)")

    s = sub.add_parser("simulate")
    s.add_argument("plan_csv")
    s.add_argument("--out", required=True)
    s.add_argument("--noise-mm", type=float, default=0.02)
    s.add_argument("--backlash-deg", type=float, nargs=2, default=(0.1, 0.1))
    s.add_argument("--seed", type=int, default=0)

    z = sub.add_parser("analyze")
    z.add_argument("csv")
    z.add_argument("--conditions", default="")
    z.add_argument("--out", help="markdown report (default: stdout)")
    z.add_argument("--json")
    a = ap.parse_args(argv)

    if a.cmd == "plan":
        geom, measured = _geometry(a.arm_calibration)
        rows = plan(tuple(a.rect), geom, a.elbow, a.cycles, a.vap_cycles, approach_mm=a.approach_mm)
        write_rows(a.out, rows)
        print("Wrote %d moves to %s%s." % (len(rows), a.out, "" if measured else
              " (joint angles from NOMINAL geometry: pass --arm-calibration once the arm is calibrated)"))
        return 0
    if a.cmd == "simulate":
        made_up = ArmGeometry((190.4, 39.7), 150.3, 179.8, (0.05, -0.04), (1.0, 1.0))
        rows = simulate(read_rows(a.plan_csv), made_up, ArmGeometry(), a.noise_mm, tuple(a.backlash_deg), a.seed)
        write_rows(a.out, rows)
        print("Wrote SYNTHETIC measurements to %s (made-up arm, noise %.3f mm, backlash %s deg)." % (a.out, a.noise_mm, a.backlash_deg))
        return 0
    res = analyze(read_rows(a.csv))
    md = to_markdown(res, a.conditions)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8", newline="\n")
        print("Wrote %s" % a.out)
    else:
        print(md)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
