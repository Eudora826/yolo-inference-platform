"""Tests for the tracing layer.

Two things matter here and they pull in opposite directions:

1. With no collector configured the module must be completely inert, so the
   image still runs fine under a bare `docker run`.
2. With a provider installed, the request must produce the child spans that
   make a trace worth looking at — a single "POST /predict took 412 ms" span
   is a black box and tells you nothing.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app import tracing
from app.main import app

client = TestClient(app)

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = (ROOT / "cat.jpg").read_bytes()


@pytest.fixture(scope="module")
def spans():
    """Capture spans in memory instead of shipping them to a collector.

    SimpleSpanProcessor, not BatchSpanProcessor: batching is asynchronous and
    a test would finish before the spans were exported. (That is also why a
    short-lived script needs an explicit force_flush() to see its own spans.)
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


def test_disabled_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    # Returns False and installs nothing: opentelemetry-api hands out
    # non-recording spans, so every tracing call becomes a no-op.
    assert tracing.setup() is False


def test_predict_splits_into_child_spans(spans):
    spans.clear()
    response = client.post(
        "/predict", files={"file": ("cat.jpg", SAMPLE, "image/jpeg")}
    )
    assert response.status_code == 200

    names = [span.name for span in spans.get_finished_spans()]
    assert "image.decode" in names
    assert "model.inference" in names


def test_annotated_endpoint_adds_its_own_span(spans):
    spans.clear()
    client.post("/predict/annotated", files={"file": ("cat.jpg", SAMPLE, "image/jpeg")})

    names = [span.name for span in spans.get_finished_spans()]
    # The extra cost of this endpoint over /predict is exactly this span.
    assert "image.annotate" in names


def test_inference_span_carries_diagnostic_attributes(spans):
    spans.clear()
    client.post("/predict", files={"file": ("cat.jpg", SAMPLE, "image/jpeg")})

    span = next(s for s in spans.get_finished_spans() if s.name == "model.inference")
    # High-cardinality values are fine on a span and fatal on a metric label:
    # a span is a discrete event, not an aggregated time series.
    assert span.attributes["image.width"] > 0
    assert span.attributes["image.height"] > 0
    assert span.attributes["model.name"] == "yolov8n.pt"
    assert "detections.count" in span.attributes


def test_video_job_gets_one_span_not_one_per_frame(spans, tmp_path, monkeypatch):
    import cv2
    import numpy as np

    monkeypatch.setenv("JOBS_DIR", str(tmp_path / "jobs"))
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64))
    for i in range(6):
        writer.write(np.full((64, 64, 3), i * 30, dtype=np.uint8))
    writer.release()

    spans.clear()
    with open(path, "rb") as fh:
        client.post("/videos", files={"file": ("tiny.mp4", fh.read(), "video/mp4")})

    job_spans = [s for s in spans.get_finished_spans() if s.name == "video.job"]
    assert len(job_spans) == 1
    assert job_spans[0].attributes["job.frames"] == 6
    assert job_spans[0].attributes["job.status"] == "completed"

    # A span per frame would mean 6 here, and thousands for a real clip —
    # enough to make the Jaeger UI unusable for no extra information.
    assert not any(s.name == "video.frame" for s in spans.get_finished_spans())
