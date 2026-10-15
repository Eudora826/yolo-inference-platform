import io
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np
import ultralytics
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app import config, jobs, metrics, tracing
from app.detector import Detector
from app.drawing import draw_detections
from app.schemas import (
    HealthResponse,
    JobCreated,
    JobInfo,
    ModelInfo,
    PredictResponse,
)

logger = logging.getLogger("yolo-service")
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(
    title="YOLO Inference Service",
    description="Object detection as a service: single images, annotated "
    "previews, and async video jobs on top of YOLOv8.",
    version="1.0.0",
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Lazy-loaded singleton: /health stays fast and tests can mock the detector.
_detector = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        # Loading the weights is the slowest thing this process ever does, and
        # it happens once, on whichever unlucky request arrives first. Worth a
        # gauge: it explains that one outlier in the latency histogram.
        t0 = time.perf_counter()
        _detector = Detector(config.model_path())
        metrics.MODEL_LOAD_SECONDS.set(time.perf_counter() - t0)
        metrics.set_model_info(
            os.path.basename(_detector.model_path),
            ultralytics.__version__,
            str(_detector.model.device or "cpu"),
        )
    return _detector


# ---------------------------------------------------------------------------
# Middleware: request id, structured access log, metrics, rate limiting
# ---------------------------------------------------------------------------

# Sliding-window request timestamps per client, for the optional rate limiter.
_hits: Dict[str, Deque[float]] = defaultdict(deque)

RATE_LIMITED_PREFIXES = ("/predict", "/videos")


def _rate_limited(client: str, path: str) -> bool:
    limit = config.rate_limit_per_minute()
    if not limit or not path.startswith(RATE_LIMITED_PREFIXES):
        return False
    now = time.monotonic()
    window = _hits[client]
    while window and now - window[0] > 60.0:
        window.popleft()
    if len(window) >= limit:
        return True
    window.append(now)
    return False


def _unrouted_path(path: str) -> str:
    """Bounded label value for requests we reject before routing.

    The rate limiter answers before FastAPI has matched a route, so
    request.scope["route"] does not exist yet and we cannot use the route
    template. The raw path is not safe to use as a label: the prefix check
    below also matches /predict/<anything>, so a client hammering random
    URLs would create an unbounded number of time series.
    """
    for prefix in RATE_LIMITED_PREFIXES:
        if path.startswith(prefix):
            return prefix
    return "other"


@app.middleware("http")
async def observability(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    client = request.client.host if request.client else "unknown"
    if _rate_limited(client, request.url.path):
        # Throttled requests are counted like any other. Leaving them out
        # would hide a real failure mode: from the dashboard's point of view
        # the traffic would simply vanish.
        throttled_path = _unrouted_path(request.url.path)
        metrics.RATE_LIMITED.labels(throttled_path).inc()
        metrics.REQUEST_COUNT.labels(request.method, throttled_path, 429).inc()
        return Response(
            content=json.dumps({"detail": "rate limit exceeded"}),
            status_code=429,
            media_type="application/json",
            headers={"Retry-After": "60", "X-Request-ID": request_id},
        )

    t0 = time.perf_counter()
    # track_inprogress() is inc()/dec() wrapped in try/finally. Doing it by
    # hand is how gauges end up stuck at a non-zero value forever: one
    # exception on the way out and the decrement never runs.
    with metrics.REQUESTS_IN_FLIGHT.track_inprogress():
        response = await call_next(request)
    duration = time.perf_counter() - t0

    # Use the route template (e.g. /jobs/{job_id}) to keep metric cardinality low.
    route = request.scope.get("route")
    if route is not None:
        path = route.path
    else:
        # Nothing matched, so this is a 404 and there is no template to use.
        # The raw URL is not safe as a label: any scanner walking /admin,
        # /.env, /wp-login.php mints a time series that is never reclaimed.
        # That 404s happened is worth counting; which URLs they were is the
        # access log's job, and the log line below still records the real path.
        path = "__unmatched__"
    metrics.REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
    metrics.REQUEST_LATENCY.labels(request.method, path).observe(duration)

    response.headers["X-Request-ID"] = request_id
    logger.info(json.dumps({
        "request_id": request_id,
        # Joins this log line to the full trace in Jaeger. None when tracing
        # is off, which is the common case for a bare `docker run`.
        "trace_id": tracing.current_trace_id(),
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration_ms": round(duration * 1000, 2),
        "client": client,
    }))
    return response


# Installed *after* the middleware above on purpose. Starlette makes the most
# recently added middleware the outermost one, so this puts the OTel server
# span around our own middleware — which is what lets the access log above
# read a valid trace_id. Swap the order and the span is already closed by the
# time we log.
tracing.setup(app)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """If API_KEY is configured, inference endpoints require it in X-API-Key."""
    expected = config.api_key()
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


# ---------------------------------------------------------------------------
# Request parsing helpers
# ---------------------------------------------------------------------------


async def read_upload(file: UploadFile, max_mb: int, kind: str) -> bytes:
    data = await file.read()
    # Recorded before the size check, so rejected uploads still show up: a
    # burst of oversized files is a signal, not something to hide.
    metrics.UPLOAD_BYTES.labels(kind).observe(len(data))
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="file too large (limit {} MB)".format(max_mb),
        )
    return data


def decode_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        # `from exc` keeps the original traceback attached, so a real decoding
        # bug is still debuggable in the logs even though the client only
        # sees a flat 415.
        raise HTTPException(
            status_code=415, detail="file is not a valid image"
        ) from exc


def parse_classes(classes: Optional[str]) -> Optional[List[int]]:
    """Turn 'person,dog' into model class ids; 400 on unknown names."""
    if not classes:
        return None
    names = [c.strip() for c in classes.split(",") if c.strip()]
    try:
        return get_detector().class_ids_for(names)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


CONF = Query(0.25, ge=0.0, le=1.0, description="confidence threshold")
IOU = Query(0.45, ge=0.0, le=1.0, description="NMS IoU threshold")
CLASSES = Query(None, description="comma-separated class names, e.g. person,dog")
MAX_DET = Query(100, ge=1, le=300, description="maximum detections per image")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    detector = get_detector()
    return {
        "model_name": os.path.basename(detector.model_path),
        "task": "detect",
        "num_classes": len(detector.names),
        "class_names": detector.names,
        "ultralytics_version": ultralytics.__version__,
        "device": str(detector.model.device or "cpu"),
    }


@app.post("/predict", response_model=PredictResponse,
          dependencies=[Depends(require_api_key)])
async def predict(
    request: Request,
    file: UploadFile = File(...),
    conf: float = CONF,
    iou: float = IOU,
    classes: Optional[str] = CLASSES,
    max_det: int = MAX_DET,
):
    data = await read_upload(file, config.max_upload_mb(), "image")
    # The automatic FastAPI instrumentation gives one span for the whole
    # request, which is a black box. These child spans are where the value
    # is: they split the number the latency histogram reports.
    with tracing.span("image.decode", **{"upload.bytes": len(data)}):
        image = decode_image(data)
    class_ids = parse_classes(classes)

    with tracing.span(
        "model.inference",
        **{
            "model.name": os.path.basename(get_detector().model_path),
            "model.conf_threshold": conf,
            "model.iou_threshold": iou,
            "image.width": image.width,
            "image.height": image.height,
        },
    ) as current:
        detections, elapsed_ms = get_detector().predict(
            image, conf=conf, iou=iou, classes=class_ids, max_det=max_det
        )
        current.set_attribute("detections.count", len(detections))
    metrics.observe_inference(detections, elapsed_ms, "image")

    return {
        "request_id": request.state.request_id,
        "model_version": os.path.basename(get_detector().model_path),
        "inference_time_ms": round(elapsed_ms, 2),
        "image_size": {"width": image.width, "height": image.height},
        "count": len(detections),
        "detections": detections,
    }


@app.post("/predict/annotated", dependencies=[Depends(require_api_key)],
          responses={200: {"content": {"image/jpeg": {}}}},
          response_class=Response)
async def predict_annotated(
    file: UploadFile = File(...),
    conf: float = CONF,
    iou: float = IOU,
    classes: Optional[str] = CLASSES,
    max_det: int = MAX_DET,
):
    """Same as /predict, but returns the image with boxes drawn (JPEG)."""
    data = await read_upload(file, config.max_upload_mb(), "image")
    with tracing.span("image.decode", **{"upload.bytes": len(data)}):
        image = decode_image(data)
    class_ids = parse_classes(classes)

    with tracing.span(
        "model.inference",
        **{
            "model.name": os.path.basename(get_detector().model_path),
            "model.conf_threshold": conf,
            "image.width": image.width,
            "image.height": image.height,
        },
    ) as current:
        detections, elapsed_ms = get_detector().predict(
            image, conf=conf, iou=iou, classes=class_ids, max_det=max_det
        )
        current.set_attribute("detections.count", len(detections))
    metrics.observe_inference(detections, elapsed_ms, "image")

    # This endpoint's extra cost over /predict lives here. Without its own
    # span you would only see that /predict/annotated is slower, not why.
    with tracing.span("image.annotate", **{"detections.count": len(detections)}):
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        draw_detections(frame, detections, {})
        ok, jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(status_code=500, detail="could not encode result")

    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Detection-Count": str(len(detections)),
            "X-Inference-Time-Ms": str(round(elapsed_ms, 2)),
        },
    )


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


@app.post("/videos", response_model=JobCreated, status_code=202,
          dependencies=[Depends(require_api_key)])
async def submit_video(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    conf: float = CONF,
    iou: float = IOU,
    classes: Optional[str] = CLASSES,
    max_det: int = MAX_DET,
):
    """Upload a video for background annotation; poll /jobs/{job_id}."""
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="unsupported video format (use {})".format(
                ", ".join(sorted(VIDEO_SUFFIXES))
            ),
        )
    data = await read_upload(file, config.max_video_upload_mb(), "video")
    class_ids = parse_classes(classes)

    job = jobs.create_job(data, suffix=suffix)
    background.add_task(
        jobs.process_job, job.job_id, get_detector(), conf, iou, class_ids, max_det
    )
    return {
        "job_id": job.job_id,
        "status": job.status,
        "status_url": "/jobs/{}".format(job.job_id),
    }


def _job_or_404(job_id: str) -> jobs.Job:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return job


@app.get("/jobs/{job_id}", response_model=JobInfo)
def job_status(job_id: str):
    job = _job_or_404(job_id)
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "frames_done": job.frames_done,
        "frames_total": job.frames_total,
        "error": job.error,
        "result_url": "/jobs/{}/result".format(job.job_id)
        if job.status == "completed" else None,
    }


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = _job_or_404(job_id)
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail="job is not completed (status: {})".format(job.status),
        )
    return FileResponse(
        job.output_path, media_type="video/mp4", filename="annotated.mp4"
    )


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics():
    return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)
