"""Prometheus metrics for the service.

Prometheus scrapes ``/metrics`` on this process directly, so every number here
is available whether the container runs alone (``docker run``) or inside the
full observability stack. Tracing is the only signal that needs a collector.

Bucket boundaries are set per instrument rather than left at the library
defaults. The defaults assume web latencies measured in whole seconds, while
yolov8n inference on this machine lands in the 58-99 ms band (20 samples,
p50 73 ms, p95 86 ms). With the default boundaries every one of those samples
falls between le=0.075 and le=0.1, so histogram_quantile has nothing to
interpolate over and reports a p95 of 97 ms for a true value of 86 ms.
See docs/metrics-design.md for the raw measurements.
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

# ---------------------------------------------------------------------------
# HTTP layer — the R, E and D of the RED method
# ---------------------------------------------------------------------------

# Errors are not a separate counter: `status` is already a label here, so
# {status=~"5.."} filters them out, and the ratio to the unfiltered total is
# the error rate. One source of truth beats two that can drift apart.
REQUEST_COUNT = Counter(
    "http_requests_total",
    "HTTP requests, by method, route template and status code",
    ["method", "path", "status"],
)

# Whole-request latency: upload, decode, inference, encode, serialise.
# Wider and more skewed than inference alone, and it has to cover video
# uploads and result downloads too, hence the 10 s ceiling.
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "End-to-end server-side request latency",
    ["method", "path"],
    buckets=(0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Concurrency is invisible in latency percentiles until it is already a
# problem: a request that is queued behind others still reports its own
# duration. Only a gauge shows the pile-up itself.
REQUESTS_IN_FLIGHT = Gauge(
    "http_requests_in_flight",
    "Requests currently being served",
)

RATE_LIMITED = Counter(
    "rate_limited_requests_total",
    "Requests rejected by the per-client rate limiter",
    ["path"],
)

# Every upload we receive, including ones we then reject as too large — a
# spike of oversized uploads is exactly what you want to be able to see.
UPLOAD_BYTES = Histogram(
    "upload_size_bytes",
    "Size of uploaded files",
    ["kind"],  # image | video
    buckets=(64e3, 256e3, 1e6, 4e6, 16e6, 64e6, 256e6),
)

# ---------------------------------------------------------------------------
# Model layer
# ---------------------------------------------------------------------------

# Measured separately from REQUEST_LATENCY on purpose: the gap between the two
# is decode + serialise cost. If p95 doubles, the two curves say immediately
# whether the model got slower or the images got bigger.
#
# `source` separates single-image requests from video frames — the same model
# call, but video frames arrive in bursts of hundreds and would otherwise
# swamp the interactive traffic in the same histogram.
INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Time inside the model for one image or video frame",
    ["source"],  # image | video_frame
    buckets=(
        0.01, 0.025,                        # headroom if we ever go GPU
        0.05, 0.06, 0.07, 0.08, 0.09, 0.1,  # the measured 58-99 ms band
        0.15, 0.25, 0.5, 1.0, 2.0,          # containers, contention, bigger models
    ),
)

# COCO has exactly 80 classes, so this label is bounded and stable. It would
# be a different decision for a model with 10k classes.
DETECTIONS = Counter(
    "detections_total",
    "Objects detected, by class",
    ["class_name"],
)

# A drift sentinel. If this normally sits at 3-5 objects per image and drops
# to 0.2, something upstream changed: a broken camera, a swapped model file,
# a misconfigured threshold. Cheap to collect, and it catches problems that
# latency and error rate never will.
DETECTIONS_PER_IMAGE = Histogram(
    "detections_per_image",
    "Objects detected in a single image or video frame",
    buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 100),  # max_det caps at 300
)

# The other half of the drift signal: confidence sliding downwards means the
# model is getting less sure about live data than it was about training data.
DETECTION_CONFIDENCE = Histogram(
    "detection_confidence",
    "Confidence score of each returned detection",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)

# Happens once, on the first request that needs the model. A gauge, not a
# counter: it is a measured value that gets set, not an accumulating total.
MODEL_LOAD_SECONDS = Gauge(
    "model_load_duration_seconds",
    "Seconds spent loading the weights (0 until the first inference)",
)

# Info metrics render as a constant 1 carrying labels — the idiomatic way to
# expose static facts. Lets a dashboard show which model version produced the
# numbers next to it.
MODEL_INFO = Info("model", "Loaded detection model")

# ---------------------------------------------------------------------------
# Async video jobs
#
# These need their own health model. POST /videos returns 202 in a few
# milliseconds and the work happens afterwards, so HTTP latency says nothing
# about it and a job that fails never produces a 5xx anywhere.
# ---------------------------------------------------------------------------

# Terminal states only. "queued" and "processing" are transient and belong in
# the gauge below; a counter can only ever go up, so it cannot represent them.
VIDEO_JOBS = Counter(
    "video_jobs_total",
    "Video annotation jobs that reached a terminal state",
    ["status"],  # completed | failed
)

# The backlog. Invisible in every other metric.
VIDEO_JOBS_ACTIVE = Gauge(
    "video_jobs_active",
    "Video jobs currently queued or processing",
)

VIDEO_JOB_DURATION = Histogram(
    "video_job_duration_seconds",
    "Wall-clock time to annotate a whole video",
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800),
)

# Throughput in frames, not jobs: a 10 s clip and a 10 min clip are not
# comparable units of work.
VIDEO_FRAMES = Counter(
    "video_frames_processed_total",
    "Frames annotated across all video jobs",
)


# ---------------------------------------------------------------------------
# Helpers — keep the instrumentation out of the request handlers
# ---------------------------------------------------------------------------


def observe_inference(detections, elapsed_ms: float, source: str) -> None:
    """Record one model call: latency, detection count, per-class counters.

    Called from both /predict and /predict/annotated and from each video
    frame, so the three endpoints cannot drift apart in what they record.
    """
    INFERENCE_LATENCY.labels(source).observe(elapsed_ms / 1000.0)
    DETECTIONS_PER_IMAGE.observe(len(detections))
    for det in detections:
        DETECTIONS.labels(det["class"]).inc()
        DETECTION_CONFIDENCE.observe(det["confidence"])


def set_model_info(model_name: str, ultralytics_version: str, device: str) -> None:
    """Publish static facts about the loaded model. Values must be strings."""
    MODEL_INFO.info({
        "model_name": model_name,
        "ultralytics_version": ultralytics_version,
        "device": device,
    })


def render() -> bytes:
    return generate_latest()


CONTENT_TYPE = CONTENT_TYPE_LATEST
