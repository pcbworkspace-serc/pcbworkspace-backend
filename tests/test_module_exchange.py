import pytest
from flask import Flask
from module_exchange import create_plan, simulate, SCENARIOS, ExchangeError
from routes_exchange import create_exchange_blueprint


def request_data(scenario="nominal"):
    return dict(module_id="dummy_module", source_id="demo_source",
                destination_id="demo_destination", scenario=scenario)


@pytest.fixture
def exchange_client():
    app = Flask(__name__)
    app.register_blueprint(create_exchange_blueprint(lambda fn: fn))
    return app.test_client()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_restraint_and_fault_gating(scenario):
    result = simulate(create_plan(request_data(scenario)))
    assert result["continuous_restraint"]
    assert result["hardware_executed"] is False
    assert result["success"] is (scenario == "nominal")
    for row in result["trace"]:
        if not row["source_retained"] and not row["destination_retained"]:
            assert row["gripper_retained"]
        if row["action"] == "gripper_release_confirmed":
            assert row["destination_retained"] and row["destination_seated"]
    if scenario != "nominal":
        assert result["trace"][-1]["action"] == "stop_preserve_restraint"
        assert not any(e["action"] == "gripper_release_confirmed" for e in result["trace"])


@pytest.mark.parametrize("data", [None, [], {}, request_data() | {"pose": [1, 2, 3]},
    request_data() | {"module_id": "real_module"}, request_data() | {"source_id": "unknown"},
    request_data() | {"scenario": True}, request_data() | {"scenario": "unknown"}])
def test_bad_request_rejected(data):
    with pytest.raises(ExchangeError):
        create_plan(data)


def test_preview_then_approved_api_exchange(exchange_client):
    data = request_data()
    preview = exchange_client.post("/exchange/plan", json=data).get_json()["plan"]
    result = exchange_client.post("/exchange/simulate", json={"request": data,
        "plan_id": preview["plan_id"], "approved": True})
    assert result.status_code == 200
    assert result.get_json()["state"] == "completed"
    assert result.get_json()["hardware_executed"] is False


@pytest.mark.parametrize("approved", [False, "true", 1, None])
def test_explicit_boolean_approval(exchange_client, approved):
    data = request_data()
    plan = create_plan(data)
    assert exchange_client.post("/exchange/simulate", json={"request": data,
        "plan_id": plan["plan_id"], "approved": approved}).status_code == 400


def test_changed_scenario_invalidates_approval(exchange_client):
    plan = create_plan(request_data())
    assert exchange_client.post("/exchange/simulate", json={
        "request": request_data("power_loss"), "plan_id": plan["plan_id"],
        "approved": True}).status_code == 409


def test_routes_use_supplied_auth_decorator():
    from functools import wraps
    def deny(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return {"error": "unauthorized"}, 401
        return wrapper
    app = Flask(__name__)
    app.register_blueprint(create_exchange_blueprint(deny))
    c = app.test_client()
    assert c.get("/exchange/capabilities").status_code == 401
    assert c.post("/exchange/plan", json=request_data()).status_code == 401
    assert c.post("/exchange/simulate", json={}).status_code == 401
