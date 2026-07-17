"""BACKEND_API_KEY auth — off by default so a deploy doesn't silently lock
out the current frontend, on and enforced once the env var is set. See the
comment above require_api_key in flask_server.py for the rollout note."""
import os


def test_auth_disabled_by_default(client):
    r = client.post("/schematic/drc", json={"circuit": {"components": []}})
    assert r.status_code == 200


def test_auth_enforced_once_key_is_configured(app):
    os.environ["BACKEND_API_KEY"] = "test-secret-123"
    try:
        client = app.app.test_client()

        r = client.post("/schematic/drc", json={"circuit": {"components": []}})
        assert r.status_code == 401

        r = client.post("/schematic/drc", json={"circuit": {"components": []}},
                         headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401

        r = client.post("/schematic/drc", json={"circuit": {"components": []}},
                         headers={"X-API-Key": "test-secret-123"})
        assert r.status_code == 200
    finally:
        os.environ.pop("BACKEND_API_KEY", None)


def test_read_only_routes_stay_open_even_with_auth_enabled(app):
    os.environ["BACKEND_API_KEY"] = "test-secret-123"
    try:
        client = app.app.test_client()
        assert client.get("/health").status_code == 200
        assert client.get("/calibration/get").status_code in (200, 404)
    finally:
        os.environ.pop("BACKEND_API_KEY", None)


def test_cors_preflight_not_blocked_by_auth(app):
    os.environ["BACKEND_API_KEY"] = "test-secret-123"
    try:
        client = app.app.test_client()
        r = client.options("/vla/plan")
        assert r.status_code == 200
    finally:
        os.environ.pop("BACKEND_API_KEY", None)
