import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from api.labels import CLASS_NAMES


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _tiny_png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 80, 40)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "device" in body


def test_classify_png(client):
    r = client.post(
        "/classify",
        files={"file": ("x.png", _tiny_png_bytes(), "image/png")},
    )
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {
        "category",
        "confidence",
        "class_index",
        "probabilities",
    }
    assert data["category"] in CLASS_NAMES
    assert data["class_index"] == CLASS_NAMES.index(data["category"])
    assert len(data["probabilities"]) == len(CLASS_NAMES)
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-5


def test_classify_rejects_non_image(client):
    r = client.post(
        "/classify",
        files={"file": ("x.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 415
