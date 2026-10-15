from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)

ROOT = Path(__file__).resolve().parent.parent
# cat.jpg is committed, so these tests always run. They used to depend on a
# 2 MB detection_example.mp4 that is not in the repo, which meant four of
# them were permanently skipped — and a test that always skips is a test
# that does not exist.
SAMPLE_IMAGE = ROOT / "cat.jpg"


def sample_bytes() -> bytes:
    return SAMPLE_IMAGE.read_bytes()


def sample_size() -> tuple:
    with Image.open(SAMPLE_IMAGE) as img:
        return img.size  # (width, height)


def post_predict(jpeg: bytes, path: str = "/predict", **params):
    return client.post(
        path,
        params=params,
        files={"file": ("frame.jpg", jpeg, "image/jpeg")},
    )


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_request_id_header():
    response = client.get("/health")
    assert "x-request-id" in response.headers


def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "yolov8n.pt"
    assert body["num_classes"] == len(body["class_names"])
    assert "person" in body["class_names"].values()


def test_index_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "YOLO Inference Service" in response.text


def test_predict_response_shape():
    width, height = sample_size()
    response = post_predict(sample_bytes())
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(body["detections"])
    assert body["model_version"] == "yolov8n.pt"
    assert body["inference_time_ms"] > 0
    assert body["image_size"] == {"width": width, "height": height}
    assert body["request_id"] == response.headers["x-request-id"]
    for det in body["detections"]:
        assert set(det) == {"class", "confidence", "bbox"}
        assert len(det["bbox"]) == 4


def test_sample_image_actually_detects_something():
    # Guards the assertions below: a filter test passes vacuously when the
    # image yields no detections at all.
    assert post_predict(sample_bytes()).json()["count"] >= 1


def test_predict_conf_threshold_is_applied():
    response = post_predict(sample_bytes(), conf=0.9)
    assert response.status_code == 200
    assert all(d["confidence"] >= 0.9 for d in response.json()["detections"])


def test_predict_class_filter():
    kept = post_predict(sample_bytes(), classes="cat")
    assert kept.status_code == 200
    assert all(d["class"] == "cat" for d in kept.json()["detections"])

    # Filtering to a class the image does not contain must return nothing,
    # rather than silently ignoring the filter.
    dropped = post_predict(sample_bytes(), classes="dog")
    assert dropped.status_code == 200
    assert dropped.json()["count"] == 0


def test_predict_unknown_class_returns_400():
    fake = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", fake)
    assert ok
    response = post_predict(jpeg.tobytes(), classes="dragon")
    assert response.status_code == 400
    assert "dragon" in response.json()["detail"]


def test_predict_rejects_non_image():
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"definitely not an image", "text/plain")},
    )
    assert response.status_code == 415


def test_predict_invalid_conf_rejected():
    response = client.post(
        "/predict",
        params={"conf": 2.0},
        files={"file": ("frame.jpg", b"x", "image/jpeg")},
    )
    assert response.status_code == 422


def test_predict_annotated_returns_jpeg():
    width, height = sample_size()
    response = post_predict(sample_bytes(), path="/predict/annotated")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert int(response.headers["x-detection-count"]) >= 0
    decoded = cv2.imdecode(
        np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None and decoded.shape == (height, width, 3)
