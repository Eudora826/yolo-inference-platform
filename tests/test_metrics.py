"""Tests for the instrumentation itself.

Instrumentation is uniquely easy to break during a refactor: it does not
affect behaviour, so nothing fails when it silently stops recording. These
tests are the only thing standing between a working dashboard and one that
quietly reports zeros.

Every assertion here is a **delta**, never an absolute value: the Prometheus
registry is process-global, so tests share it and run in an arbitrary order.
"""

from pathlib import Path

from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app import main as service

client = TestClient(service.app)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = (ROOT / "cat.jpg").read_bytes()


def value(name: str, labels: dict = None) -> float:
    """Current value of one sample, 0.0 if it does not exist yet.

    A labelled Counter does not appear in /metrics until the first time that
    exact label combination is observed, so `or 0.0` is not optional.
    Labels must match *exactly* — a missing one returns None, not a subtotal.
    """
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def image_file():
    return {"file": ("cat.jpg", SAMPLE, "image/jpeg")}


# --- HTTP layer ------------------------------------------------------------


def test_request_counter_increments_per_request():
    labels = {"method": "GET", "path": "/health", "status": "200"}
    before = value("http_requests_total", labels)
    for _ in range(3):
        client.get("/health")
    assert value("http_requests_total", labels) == before + 3


def test_client_errors_are_counted_under_their_own_status():
    labels = {"method": "POST", "path": "/predict", "status": "415"}
    before = value("http_requests_total", labels)
    client.post("/predict", files={"file": ("x.txt", b"nope", "text/plain")})
    assert value("http_requests_total", labels) == before + 1


def test_in_flight_gauge_returns_to_zero():
    # If this ever fails, the gauge is leaking on some code path and the
    # "concurrency" panel is showing a number that only ever goes up.
    client.get("/health")
    assert value("http_requests_in_flight") == 0.0


def test_rejected_uploads_are_still_measured():
    before = value("upload_size_bytes_count", {"kind": "image"})
    client.post("/predict", files={"file": ("x.txt", b"nope", "text/plain")})
    # Recorded before the format check on purpose: a flood of junk uploads
    # is a signal worth seeing, not something to hide.
    assert value("upload_size_bytes_count", {"kind": "image"}) == before + 1


def test_rate_limited_requests_are_counted_twice_over(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    service._hits.clear()

    rl_labels = {"path": "/predict"}
    http_labels = {"method": "POST", "path": "/predict", "status": "429"}
    before_rl = value("rate_limited_requests_total", rl_labels)
    before_429 = value("http_requests_total", http_labels)

    statuses = [
        client.post("/predict", files={"file": ("x.jpg", b"x", "image/jpeg")}).status_code
        for _ in range(3)
    ]
    assert statuses == [415, 429, 429]

    # Two separate metrics on purpose: the dedicated counter answers "is the
    # limiter firing", the 429s keep the request total honest so traffic
    # does not appear to vanish from the dashboard.
    assert value("rate_limited_requests_total", rl_labels) == before_rl + 2
    assert value("http_requests_total", http_labels) == before_429 + 2

    service._hits.clear()


# --- Model layer -----------------------------------------------------------


def test_inference_metrics_track_requests():
    labels = {"source": "image"}
    before_count = value("inference_duration_seconds_count", labels)
    before_sum = value("inference_duration_seconds_sum", labels)

    client.post("/predict", files=image_file())

    assert value("inference_duration_seconds_count", labels) == before_count + 1
    assert value("inference_duration_seconds_sum", labels) > before_sum


def test_detections_are_counted_per_class():
    before = value("detections_total", {"class_name": "cat"})
    client.post("/predict", files=image_file())
    assert value("detections_total", {"class_name": "cat"}) > before


def test_annotated_endpoint_records_the_same_model_metrics():
    # /predict and /predict/annotated share one helper precisely so they
    # cannot drift apart. This test is what keeps that true.
    labels = {"source": "image"}
    before = value("inference_duration_seconds_count", labels)
    client.post("/predict/annotated", files=image_file())
    assert value("inference_duration_seconds_count", labels) == before + 1


def test_model_load_time_is_recorded():
    client.post("/predict", files=image_file())  # forces the lazy load
    assert value("model_load_duration_seconds") > 0


# --- Async video jobs ------------------------------------------------------


def test_failed_job_counts_and_balances_the_gauge(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_DIR", str(tmp_path / "jobs"))

    before_failed = value("video_jobs_total", {"status": "failed"})
    before_active = value("video_jobs_active")
    before_duration = value("video_job_duration_seconds_count")

    # A file that is named .mp4 but is not a video: OpenCV cannot open it,
    # process_job() raises, and we take the except branch.
    response = client.post(
        "/videos", files={"file": ("broken.mp4", b"not really a video", "video/mp4")}
    )
    assert response.status_code == 202
    # TestClient runs BackgroundTasks synchronously, so the job is finished
    # by the time the response comes back.
    status = client.get("/jobs/{}".format(response.json()["job_id"])).json()
    assert status["status"] == "failed"

    assert value("video_jobs_total", {"status": "failed"}) == before_failed + 1
    assert value("video_job_duration_seconds_count") == before_duration + 1

    # The important one. Without the finally: block this would be
    # before_active + 1, and every failed job would inflate the backlog
    # panel forever — a fake backlog looks exactly like a real one.
    assert value("video_jobs_active") == before_active


def test_successful_job_records_frames(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    monkeypatch.setenv("JOBS_DIR", str(tmp_path / "jobs"))
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64))
    for i in range(6):
        writer.write(np.full((64, 64, 3), i * 30, dtype=np.uint8))
    writer.release()

    before_frames = value("video_frames_processed_total")
    before_completed = value("video_jobs_total", {"status": "completed"})
    before_active = value("video_jobs_active")

    with open(path, "rb") as fh:
        response = client.post(
            "/videos", files={"file": ("tiny.mp4", fh.read(), "video/mp4")}
        )
    assert response.status_code == 202

    assert value("video_frames_processed_total") == before_frames + 6
    assert value("video_jobs_total", {"status": "completed"}) == before_completed + 1
    assert value("video_jobs_active") == before_active
    # Video frames go through the same model call but land under their own
    # `source` label, so a burst of frames cannot distort the p95 that the
    # interactive /predict panel reports.
    assert value("inference_duration_seconds_count", {"source": "video_frame"}) >= 6
