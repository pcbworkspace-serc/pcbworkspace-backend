# Metrology: error budget, calibration, ISO 9283

Tools for one honest number: how accurately the real arm puts a part where the
board file says it goes, and which error source to fix first. Everything here
runs without hardware. Results from synthetic data or assumed inputs say so in
their output. Only measurements from the assembled arm can replace them.

| Step | Tool | Needs hardware? | Output |
| --- | --- | --- | --- |
| A. Error budget | `python -m metrology.error_budget` | No | [ERROR_BUDGET.md](ERROR_BUDGET.md): predicted accuracy, ranked error sources, what to measure; [ADR-001](ADR-001-joint-drive.md): the joint drive it led to |
| B1. Camera intrinsics | `python -m metrology.charuco_calib intrinsics photos/ ...` | Camera + printed target | Camera matrix, lens distortion, reprojection error (in-sample and held-out) |
| B2. Board plane | `python -m metrology.charuco_calib homography plane.jpg ...` | Camera + target | POST body for `/calibration/save`, with distortion correction and residuals |
| B3. Arm geometry | `python -m metrology.kinematic_fit plan / fit` | Assembled arm | `arm_calibration.json`: link lengths, base offset, joint zeros, with standard errors |
| C. Accuracy test | `python -m metrology.iso9283 plan / analyze` | Assembled arm | ISO 9283-style AP, RP, vAP per pose, against the team's targets |

## What the budget says now

The full table is in [ERROR_BUDGET.md](ERROR_BUDGET.md). It is a prediction from
datasheets and stated assumptions.

- **With direct-drive steppers, joint angle is the problem.** At the board's
  current placeholder position, the far corner is 265 mm from the base. There,
  0.1° at the base joint moves the nozzle 0.46 mm. The team's target is 0.15 mm RMS.
- **Calibration matters most first.** Before calibration, the unmeasured base
  offset and joint zeros dominate (about 5 mm). Steps B and C remove most of
  that.
- **After calibration, the stepper limits remain.** Load deflection (hose drag
  against a stepper at hold current), ±5% step error and backlash leave about
  0.8 mm RMS. Approaching every target from the same side brings that to about
  0.4 mm.
- **The AS5048A is not an accuracy fix on its own.** Its datasheet INL is ±0.8°
  at best magnet placement. It reaches 0.05° only after per-unit linearization,
  and even then closed loop on direct drive predicts 0.21 mm RMS. The BOM puts
  the encoders on base and shoulder, not on the elbow.
- **Decision: 50:1 strain-wave reducers on J1 and J2, open loop**
  ([ADR-001](ADR-001-joint-drive.md)). Of four drive options, this is the only
  one predicted to meet both targets: 0.129 mm RMS and 0.036 mm repeatability.
  The margin is thin, and it rests on the reducer's transmission error
  (assumed 2 arcmin). The ADR gates the second purchase on measuring the first
  reducer.

Every assumed input is listed at the bottom of the report with its path in
[budget_config.json](budget_config.json). Measure it, edit the value, change the
source, and re-run.

## B. Calibration

```
python -m metrology.charuco_calib make-board --out charuco.png
python -m metrology.charuco_calib intrinsics photos/ --square-mm 10.02 --out top_camera.json
python -m metrology.charuco_calib homography plane.jpg --intrinsics top_camera.json --out plane.json
```

- Print the target at 100% and measure a square with calipers. Pass the
  measured size, because printers scale.
- Take 15–30 photos at varied tilts that cover the image corners. The report
  shows reprojection RMS on the views used and on held-out views. A big gap
  between them means you need more or better views.
- For the plane photo, lay the target at **board-top height**. A 1.6 mm
  height error at 200 mm camera height shifts edge points by about 0.3 mm.
- `plane.json → save_request` is the body for `POST /calibration/save`. The
  endpoint now takes optional `intrinsics`, undistorts the points before
  fitting, and returns `residual` (RMS, max and held-out in mm). With only 4
  points the residual is 0 by construction, and the response says so.

Arm geometry, which replaces the `ARM_BASE_OFFSET_MM` placeholder:

```
python -m metrology.kinematic_fit plan --out cal_poses.csv
# command each row's q1/q2 (always approaching from the same side), measure the tip in board mm
python -m metrology.kinematic_fit fit cal_poses.csv --out arm_calibration.json
python -m metrology.error_budget --arm-calibration arm_calibration.json
```

Put `arm_calibration.json` in the backend root, or set `ARM_CALIBRATION_PATH`.
`layout_engine` then uses the measured base offset and link lengths, and
`/nn/status → robot_safety.kinematic_reach_check.arm_geometry` says
`measured <date>`. A calibration made with `simulate` is marked synthetic and
never loaded.

## C. ISO 9283-style test

```
python -m metrology.iso9283 plan --out iso_plan.csv --arm-calibration arm_calibration.json
# run the 271 moves; fill x_meas_mm / y_meas_mm (lead-in row needs no measurement)
python -m metrology.iso9283 analyze iso_plan.csv --conditions "50% speed, no part, 30 min warm-up, 21 C" --out iso_report.md
```

- Poses P1–P5 follow the standard's measuring plane. The plan runs 30 cycles,
  then vAP from four in-plane directions at P1, P2 and P4.
- The report has AP and RP per pose with bootstrap 95% CIs, plus vAP. It also
  has the two metrics in the team's own plan: RMS deviation and per-axis std.
- Deviations from the standard are written into the report: planar only, and
  in-plane vAP directions.
- The analysis checks the measurements against commanded positions. To check
  against the board file, feed it the KiCad `.pos` positions as commands.

## Not done here (needs the arm)

- No input in `budget_config.json` has been measured.
- The calibration tools have only been tested on rendered images of a made-up
  camera (`charuco_calib demo`) and simulated arms (`kinematic_fit simulate`,
  `iso9283 simulate`).
- End-to-end placement accuracy (step E in the plan: 20+ parts measured against
  the `.pos` file) needs the arm.
