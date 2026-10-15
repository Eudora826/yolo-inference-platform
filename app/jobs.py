"""In-memory async video jobs.

A job is created by POST /videos, processed in the background (frame by frame
through the detector, boxes drawn with app.drawing), and exposed through
GET /jobs/{id}. For a single-process service an in-memory registry is enough;
swapping in Redis/Celery would keep the same interface.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2

from app import config, metrics, tracing
from app.drawing import draw_detections

_jobs: Dict[str, Job] = {}
_lock = threading.Lock()


@dataclass
class Job:
    job_id: str
    status: str = "queued"  # queued | processing | completed | failed
    frames_done: int = 0
    frames_total: int = 0
    error: Optional[str] = None
    input_path: str = ""
    output_path: str = ""

    @property
    def progress(self) -> float:
        if self.status == "completed":
            return 1.0
        if not self.frames_total:
            return 0.0
        return round(self.frames_done / self.frames_total, 4)


def create_job(video_bytes: bytes, suffix: str = ".mp4") -> Job:
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(config.jobs_dir(), job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_path = os.path.join(job_dir, "input" + suffix)
    with open(input_path, "wb") as f:
        f.write(video_bytes)

    job = Job(job_id=job_id, input_path=input_path,
              output_path=os.path.join(job_dir, "annotated.mp4"))
    with _lock:
        _jobs[job_id] = job
    # Incremented at creation, not when processing starts, so the gauge counts
    # queued work too — the backlog is the whole point of this metric.
    # process_job() decrements it on every exit path.
    metrics.VIDEO_JOBS_ACTIVE.inc()
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def process_job(
    job_id: str,
    detector,
    conf: float,
    iou: float,
    classes: Optional[List[int]],
    max_det: int,
) -> None:
    """Annotate the job's video. Runs in the background; updates job state."""
    job = get_job(job_id)
    if job is None:
        # create_job() already counted this one; balance the gauge and leave.
        metrics.VIDEO_JOBS_ACTIVE.dec()
        return
    job.status = "processing"
    started = time.perf_counter()

    # One span for the whole job, deliberately NOT one per frame: a 250-frame
    # clip would produce 250 spans, which makes the Jaeger UI unusable and
    # tells you nothing the per-frame histogram does not already say.
    #
    # This is its own root trace, not a child of POST /videos — the background
    # task starts after the HTTP response has been sent, so that server span
    # is already closed. Relating the two properly is what OTel Span Links
    # exist for.
    with tracing.span("video.job", **{"job.id": job_id}) as current:
        try:
            cap = cv2.VideoCapture(job.input_path)
            if not cap.isOpened():
                raise RuntimeError("could not open the uploaded video")

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            job.frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(job.output_path, fourcc, fps, (width, height))
            class_ids: Dict[str, int] = {}

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                detections, frame_ms = detector.predict(
                    frame, conf=conf, iou=iou, classes=classes, max_det=max_det
                )
                metrics.observe_inference(detections, frame_ms, "video_frame")
                metrics.VIDEO_FRAMES.inc()
                draw_detections(frame, detections, class_ids)
                writer.write(frame)
                job.frames_done += 1

            cap.release()
            writer.release()
            if job.frames_done == 0:
                raise RuntimeError("no frames could be decoded from the video")
            job.status = "completed"
        except Exception as exc:  # surface the failure to the client, don't crash
            job.status = "failed"
            job.error = str(exc)
        finally:
            # Every exit path, success or failure. A gauge that is only
            # decremented on the happy path drifts upwards forever, and a
            # permanently rising backlog is indistinguishable from a real one.
            current.set_attribute("job.status", job.status)
            current.set_attribute("job.frames", job.frames_done)
            metrics.VIDEO_JOBS_ACTIVE.dec()
            metrics.VIDEO_JOBS.labels(job.status).inc()
            metrics.VIDEO_JOB_DURATION.observe(time.perf_counter() - started)
