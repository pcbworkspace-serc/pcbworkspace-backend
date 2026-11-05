"""Stateless simulation API, sharing the backend's existing auth decorator."""
from flask import Blueprint, jsonify, request
from module_exchange import create_plan, simulate, ExchangeError, REGISTRY_VERSION, SCENARIOS


def create_exchange_blueprint(auth):
    bp = Blueprint("module_exchange", __name__, url_prefix="/exchange")

    @bp.get("/capabilities")
    @auth
    def capabilities():
        return jsonify({"ok": True, "mode": "simulation", "hardware_enabled": False,
                        "registry_version": REGISTRY_VERSION, "scenarios": list(SCENARIOS)})

    @bp.post("/plan")
    @auth
    def plan():
        try:
            result = create_plan(request.get_json(silent=True))
        except ExchangeError as e:
            return jsonify({"ok": False, "error": str(e), "hardware_executed": False}), 400
        return jsonify({"ok": True, "plan": result})

    @bp.post("/simulate")
    @auth
    def run():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or set(data) != {"request", "approved", "plan_id"}:
            return jsonify({"ok": False, "error": "Provide request, approved and plan_id"}), 400
        if data["approved"] is not True:
            return jsonify({"ok": False, "error": "Explicit approval required for this simulation"}), 400
        try:
            planned = create_plan(data["request"])
        except ExchangeError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        if data["plan_id"] != planned["plan_id"]:
            return jsonify({"ok": False, "error": "Plan changed; preview and approve again"}), 409
        return jsonify(simulate(planned))

    return bp
