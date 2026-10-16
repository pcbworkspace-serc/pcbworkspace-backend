# ADR-001: Joint drive for the MiniMEE arm (J1 base, J2 elbow)

**Status:** Proposed
**Date:** 2026-10-16
**Deciders:** SERC hardware lead (sign-off), firmware owner (encoder and approach changes)

## Context

The arm is a 2-link SCARA (150 mm + 180 mm links). The documented design drives
the base (J1) and elbow (J2) joints directly from the stepper shafts: a NEMA 17
at J1 and a NEMA 14 at J2, with no reduction. Targets from hardware_plan_v3,
section 9: placement accuracy < 0.15 mm RMS, repeatability < 0.1 mm std.

The board's far corner sits 265 mm from the base, using the placeholder offset.
At that radius, every 0.1° of J1 error moves the nozzle 0.46 mm. Holding
0.15 mm RMS therefore needs about 0.02° at the joints. That is roughly a tenth
of a 16-microstep step on a direct-drive motor, and below the ±0.09° step error
on a typical stepper datasheet.

The numbers below come from the error budget (`python -m metrology.error_budget`,
full tables in [ERROR_BUDGET.md](ERROR_BUDGET.md)). Every option was evaluated
on the same footing: camera, kinematic and nozzle calibration done, every
target approached from one side, worst board point. **They are predictions.
The arm has not been built, and the inputs marked "assumed" in
[budget_config.json](budget_config.json) have not been measured.**

## Decision

Use **option D**:

1. **50:1 strain-wave (harmonic) reducers on J1 and J2**, NEMA 17 flange
   input. The links bolt to the reducer output flanges, with no set-screw or
   D-shaft joints.
2. **Open-loop stepping, with no closed-loop firmware.** The AS5048A encoders
   stay on the motor shafts, where reduction divides their ±0.8° INL by 50.
   They handle homing: an output-side home switch picks the motor turn, then
   the encoder gives the fine zero. They also detect missed steps: the firmware
   compares the encoder against the commanded steps after every move and stops
   if they disagree by more than one full step.
3. **Move the second AS5048A from the shoulder to the elbow motor.** The BOM
   puts encoders on the base and shoulder, but the elbow is the joint that
   affects placement. This assumes the shoulder NEMA 17 drives Z; confirm it.
4. **Approach every target from one side**, from the negative direction on both
   joints. The same rule applies during calibration and the ISO test.
5. **Keep 16 microsteps and the current board position.** At 50:1, one
   microstep moves the tip about 0.01 mm. Moving the board 30 mm closer to the
   base gains about 0.007 mm (distance table in ERROR_BUDGET.md), which isn't
   worth constraining feeder and camera placement.
6. **Incoming inspection gates the second purchase.** Buy one reducer first,
   measure its transmission error and lost motion, and then buy the second.
   The decision depends on this measurement (see "What would change this").

## Options considered

Accuracy is the predicted RMS at the worst board point. Repeatability is the
largest per-axis std. Both come from the error budget.

| | A: direct drive, closed loop | B: GT2 belt 4:1, open loop | C: GT2 belt 4:1, closed loop on output encoder | **D: strain-wave 50:1, open loop** |
| --- | --- | --- | --- | --- |
| Predicted accuracy RMS | 0.208 mm | 0.235 mm | 0.201 mm | **0.129 mm** |
| Predicted repeatability std | 0.095 mm | 0.036 mm | 0.095 mm | **0.036 mm** |
| Meets both targets | no | no | no | **yes, thin margin** |
| Largest error left | encoder INL after linearization (0.05°, datasheet best case) | belt pulley runout (assumed 0.02 mm) | encoder INL, now at the output so not divided | reducer transmission error (assumed 2 arcmin) |
| Extra hardware cost | ~$0 (encoder moved) | ~$20–40 per joint, pulleys and belt | B plus encoder mounts | ~$56–140 per reducer in current listings |
| Firmware | closed loop + per-unit encoder linearization | open loop | closed loop + linearization | open loop + missed-step check |
| Joint speed at 1000 rpm motor | ~6000°/s (not the limit) | ~1500°/s | ~1500°/s | **~120°/s** |
| Added mass | none | small | small | ~0.4 kg per reducer (listing weight) |
| Main risk | linearization never reaches 0.05° | belt stiffness unknown: no published GT2 figure | as A, plus output-shaft magnet mounting | clone reducer quality varies |

**A (direct drive, closed loop on linearized AS5048A).** Pros: no new
mechanical parts. Cons: the AS5048A datasheet gives ±0.8° INL at best magnet
placement. Only linearization in the MCU brings it to 0.05°, and even that
leaves 0.13 mm at J1. Closed-loop settling on one microstep adds noise, so
repeatability gets worse (0.095 mm). Firmware effort is the highest of the four.

**B (GT2 belt 4:1, open loop).** Pros: cheap, printable, zero backlash.
Cons: 4:1 only divides stepper error by 4, and pulley runout becomes
transmission error. Belt stiffness is an order-of-magnitude assumption: I
found no published figure, so it needs a tensile test. A two-stage 16:1 belt
would help the stepper terms, but not the runout of the output pulley.

**C (belt with output-side encoder, closed loop).** It corrects belt stretch
and runout, but the encoder now sits on the output. Its INL is no longer
divided by the ratio, so this lands where A does.

**D (strain-wave 50:1, open loop).** Pros: divides every motor-side error by
50, including step error, step size and encoder INL. It multiplies motor
stiffness by 2500, so hose drag stops mattering (0.0006° deflection). Lost
motion is small: Harmonic Drive's size-11 catalog gives about 2 arcmin, and
clones claim < 20 arcsec. Firmware stays open loop. Cons: slower joints, more
mass at the elbow, and cost. The result rests on the reducer's transmission
error, which isn't in the clone listings.

## Trade-off analysis

The deciding fact is where each design's dominant error sits relative to the
reduction. In A and C, the encoder that closes the loop sets the floor, and its
INL reaches the joint undivided (A is direct drive, C measures at the output).
In B, the floor is the output pulley's runout. Only D puts almost every
error source on the motor side, where 50:1 shrinks it. What remains on the
output side is the reducer's own transmission error, which is the one number
to verify.

D costs speed. At 120°/s a 60° move takes roughly 0.7 s with acceleration.
That fits the low-volume, high-mix use in the hardware plan, but not a
high-throughput line.

## What would change this

From the budget's sensitivity runs on option D:

| If measured | Option D accuracy | Consequence |
| --- | --- | --- |
| TE 1 arcmin | 0.089 mm | Comfortable margin |
| TE 2 arcmin (assumed) | 0.129 mm | Meets target, thin margin |
| TE 4 arcmin | 0.227 mm | Fails, and is no better than A or C. Map the TE per joint with the top camera, or return the unit |
| Lost motion 6 arcmin | 0.139 mm | Still meets, because of one-sided approach |
| Release slip 0.06 mm instead of 0.03 | 0.148 mm | At the limit; fix the release process |
| Hose drag 5× assumed | 0.129 mm | No change at 50:1 |
| Tip measurement noise 0.06 mm | 0.131 mm | No change; calibration isn't the limit |

If the first reducer measures worse than about 3 arcmin TE and mapping doesn't
recover it, reopen the decision. The next candidates are a genuine Harmonic
Drive unit (price needs a quote; the ~$336 "CSF-style" listing found is
another third-party copy) or a capstan drive.

## Consequences

- **Easier:** firmware stays open loop. Hose drag, step error and encoder INL
  stop limiting accuracy, and the anti-backlash spring nuts are no longer
  needed at J1 and J2.
- **Harder:** the base and elbow mounts must be redesigned around the reducer
  flange. The printed elbow joint carries about 0.4 kg more, so check Z sag at
  full reach. Throughput drops.
- **To revisit:** after the first reducer's inspection, and again after the
  ISO 9283 test on the built arm. Update `budget_config.json` with every
  measured value; the budget then shows whether the decision still holds.

## Action items

1. [ ] Buy one 50:1 NEMA 17 strain-wave reducer. Lock the motor and measure
   lost motion with a dial indicator at a known radius, loading both ways.
   Then step the input through one full output turn and measure transmission
   error with the top camera on a target on the output. Enter both in
   `budget_config.json` (`joints.j1.reducer_*`) and rerun the budget.
2. [ ] If TE ≤ 2 arcmin, buy the second reducer. If not, see "What would
   change this".
3. [ ] Redesign the base and elbow mounts for the reducer flange, with links
   bolted to the output flange. Add an output-side home switch on J1 and J2.
4. [ ] Move the second AS5048A to the elbow motor. Confirm what the shoulder
   NEMA 17 drives.
5. [ ] Firmware: one-sided approach on both joints; compare the encoder with
   commanded steps after each move.
6. [ ] Build, then run `charuco_calib`, `kinematic_fit` and the `iso9283` test
   (metrology/README.md). The ISO result, not this budget, is the number for
   the SBIR reviewers.

## Sources

- AS5048A INL ±0.8° (optimum placement, 25 °C), and 0.05° system accuracy only with MCU linearization and averaging: [ams AS5048 datasheet](https://cdn.sparkfun.com/assets/e/d/e/1/d/AS5048_DS000298_4-00.pdf)
- Stepper step-angle accuracy ±5% of a full step, no load: e.g. [RS / SLAC hybrid stepper datasheet](https://confluence.slac.stanford.edu/download/attachments/456196196/StepperMotorRS535-0489.pdf?api=v2)
- Strain-wave size 11, ratio 50, stiffness K1 about 2,200 N·m/rad, lost motion about 2 arcmin: [Harmonic Drive CSF-ULW brochure](https://www.harmonicdrive.net/_hd/content/documents1/CSF-ULW%20Brochure%20HR.pdf)
- NEMA 17 strain-wave reducers, 50:1, < 20 arcsec claimed backlash, $56–140: [VXB listing](https://vxb.com/products/high-torque-harmonic-gear-drive-reducer-20-arcsec)
- GT2 belt stiffness depends on length and is non-linear; no per-metre figure found: [Robotics 7(4):75 technical note](https://www.mdpi.com/2218-6581/7/4/75/html)
