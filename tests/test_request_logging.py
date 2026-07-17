import logging


def test_request_logged_with_method_path_status_timing(client, caplog):
    with caplog.at_level(logging.INFO, logger="pcbworkspace.request"):
        client.get("/health")

    lines = [r.message for r in caplog.records if r.name == "pcbworkspace.request"]
    assert any("GET" in l and "/health" in l and "-> 200" in l and "ms)" in l for l in lines), lines


def test_401_logged_with_key_missing_marker(app, client, caplog):
    import os
    os.environ["BACKEND_API_KEY"] = "secret"
    try:
        with caplog.at_level(logging.INFO, logger="pcbworkspace.request"):
            client.post("/schematic/drc", json={"circuit": {"components": []}})
        lines = [r.message for r in caplog.records if r.name == "pcbworkspace.request"]
        assert any("-> 401" in l and "key=missing" in l for l in lines), lines
    finally:
        os.environ.pop("BACKEND_API_KEY", None)
