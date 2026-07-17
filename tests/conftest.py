"""Shared pytest fixtures — keeps CWD at the backend root so relative
paths (best.pt, jepa_heads_synthetic.pt, calibration.json) resolve the
same way they do when Render runs `python flask_server.py`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(autouse=True, scope="session")
def _chdir_to_backend_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    old = os.getcwd()
    os.chdir(root)
    yield
    os.chdir(old)


@pytest.fixture
def app():
    os.environ.pop("BACKEND_API_KEY", None)  # tests assume auth off unless a test says otherwise
    import flask_server as srv
    return srv


@pytest.fixture
def client(app):
    return app.app.test_client()
