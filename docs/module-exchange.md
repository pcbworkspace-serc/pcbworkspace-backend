# Captive-module exchange integration, first stage

This API exercises logical restraint handoffs through the backend. It has no serial
or motor access. It is not physical robot integration or microgravity validation.

`GET /exchange/capabilities` reports simulation-only capability.

`POST /exchange/plan` accepts exactly:

```json
{"module_id":"dummy_module","source_id":"demo_source","destination_id":"demo_destination","scenario":"nominal"}
```

The preview identifies the registry version, scenario, limitations and plan ID.
`POST /exchange/simulate` accepts `request` (the same object), `plan_id` and
`approved: true`. A changed request or registry invalidates the ID. This is a
simulation confirmation, not authentication or authorization for real hardware.
All routes share `require_api_key`; its existing optional BACKEND_API_KEY behavior
still applies. Do not put a shared backend credential into a public web build.

Scenarios: nominal, missing_grasp, source_stuck, destination_unlatched,
stale_feedback, power_loss. A fault produces `state: faulted` and retains the
simulated module at its last confirmed holder. Every trace row labels feedback
synthetic; `hardware_executed` is always false. Power-loss retention is an assumed
mechanical requirement in the simulation, not tested device behavior.

The matching Workspace panel uses VITE_NN_URL (the existing AI backend URL),
previews this plan, requests explicit confirmation, and displays the trace.
It does not call WebSerial, /robot/chat, PICK/PLACE, or vacuum control.

## Remaining hardware integration

Select gripper actuator/guides/passive lock and sensors. Measure the fixtures,
payload, connector loads, axis configuration and limits. Implement the actual
2R+Z+yaw kinematics and one ID-correlated serial contract. Commission physical
homing/interlocks/watchdog and power-loss Z/grip retention. Then integrate live
grasp/source/destination feedback and validate a guarded inert-module exchange.

Existing suction commands and alternative firmware protocols remain historical
paths. This addition does not repair them or enable the new gripper.
