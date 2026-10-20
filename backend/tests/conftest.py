import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="videoai-test-")
os.environ.update(
    DATABASE_URL=f"sqlite:///{_tmp}/test.db",
    STORAGE_DIR=f"{_tmp}/storage",
    WORKER_ENABLED="false",  # tests drive the worker explicitly
    TTS_PROVIDERS="synthetic",  # hermetic: no network / speech engine needed
    RENDER_SCALE="0.3",
    RENDER_FPS="12",
    SECRET_KEY="test-secret-test-secret-test-secret-123",
    RENDER_DAILY_LIMIT_USER="3",
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:  # runs lifespan => alembic migrations
        yield c


def _signup(client, email):
    r = client.post("/api/auth/signup", json={"email": email, "password": "password123"})
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture()
def auth(client):
    import uuid

    return _signup(client, f"{uuid.uuid4().hex[:8]}@example.com")


@pytest.fixture()
def other_auth(client):
    import uuid

    return _signup(client, f"{uuid.uuid4().hex[:8]}@example.com")
