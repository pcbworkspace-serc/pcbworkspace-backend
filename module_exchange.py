"""Deterministic captive-module exchange preview. No hardware or ML access.

The demonstration registry deliberately has no poses: these are logical handoff
tests, not kinematics, load, sensor, or microgravity validation.
"""
import hashlib
import json

REGISTRY_VERSION = "dummy-workcell-v1"
SCENARIOS = ("nominal", "missing_grasp", "source_stuck",
             "destination_unlatched", "stale_feedback", "power_loss")
REQUEST_FIELDS = {"module_id", "source_id", "destination_id", "scenario"}
STAGES = ("capture", "source_release", "transfer", "seat", "release")


class ExchangeError(ValueError):
    pass


def create_plan(data):
    if not isinstance(data, dict) or set(data) != REQUEST_FIELDS:
        raise ExchangeError("Provide exactly module_id, source_id, destination_id and scenario")
    if any(not isinstance(v, str) for v in data.values()):
        raise ExchangeError("All request fields must be strings")
    if data["module_id"] != "dummy_module":
        raise ExchangeError("Only dummy_module is registered for simulation")
    if data["source_id"] != "demo_source" or data["destination_id"] != "demo_destination":
        raise ExchangeError("Only demo_source to demo_destination is registered")
    if data["scenario"] not in SCENARIOS:
        raise ExchangeError("Unknown simulation scenario")
    payload = {**data, "registry_version": REGISTRY_VERSION, "mode": "simulation"}
    plan_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {**payload, "plan_id": plan_id, "stages": list(STAGES),
            "hardware_executed": False,
            "limitations": ["No actuator commands are sent",
                            "Feedback is synthetic, not measured",
                            "No kinematics, collision, force or gravity validation"]}


def simulate(plan):
    """Run one finite logical scenario; faults stop and preserve restraint.

    Consumers must use create_plan first. This function never opens a serial
    port and does not call legacy PICK/PLACE or vacuum primitives.
    """
    scenario = plan["scenario"]
    source = True
    grasp = False
    destination = False
    seated = False
    trace = []

    def event(stage, action):
        trace.append({"stage": stage, "action": action, "source_retained": source,
                      "gripper_retained": grasp, "destination_retained": destination,
                      "destination_seated": seated, "feedback_kind": "synthetic"})

    def finish(state, fault=None):
        return {"ok": True, "mode": "simulation", "hardware_executed": False,
                "plan_id": plan["plan_id"], "state": state,
                "success": state == "completed", "fault": fault, "trace": trace,
                "continuous_restraint": all(e["source_retained"] or
                    e["gripper_retained"] or e["destination_retained"] for e in trace)}

    def stop(stage, reason):
        event(stage, "stop_preserve_restraint")
        return finish("faulted", reason)

    event("capture", "capture_while_source_retains")
    if scenario == "missing_grasp":
        return stop("capture", "grasp_not_confirmed")
    grasp = True
    event("capture", "grasp_confirmed")
    if scenario == "source_stuck":
        return stop("source_release", "source_latch_not_released")
    # Receiving restraint confirmed before releasing the source.
    source = False
    event("source_release", "source_release_confirmed")
    if scenario in ("stale_feedback", "power_loss"):
        return stop("transfer", scenario)
    event("transfer", "transport_with_grasp")
    seated = True
    event("seat", "seating_confirmed")
    if scenario == "destination_unlatched":
        return stop("seat", "destination_latch_not_confirmed")
    destination = True
    event("seat", "destination_latch_confirmed")
    # Both destination conditions confirmed before releasing the gripper.
    grasp = False
    event("release", "gripper_release_confirmed")
    event("completed", "inventory_updated_in_simulation")
    return finish("completed")
