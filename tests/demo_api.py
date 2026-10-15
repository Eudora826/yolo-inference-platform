"""
Annotate a video by sending every frame through the FastAPI /predict endpoint.

This combines demo.py and test_api.py: frames are JPEG-encoded and POSTed to
the service via FastAPI's TestClient (in-process, no server needed), and the
JSON detections are drawn with the same helpers demo.py uses.

Usage (run from the repo root so `app` and `tests` are importable):
    python -m tests.demo_api detection_example.mp4 detection_example_api.mp4
    python -m tests.demo_api input.mp4 output.mp4 --max-frames 50
"""

from __future__ import annotations

import argparse
import time

import cv2
from fastapi.testclient import TestClient

from app.drawing import color_for, draw_box
from app.main import app

client = TestClient(app)


def detect_via_api(frame) -> list[dict]:
    """POST one BGR frame to /predict and return the parsed detections."""
    ok, jpeg = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    response = client.post(
        "/predict",
        files={"file": ("frame.jpg", jpeg.tobytes(), "image/jpeg")},
    )
    response.raise_for_status()
    return response.json()["detections"]


def annotate_via_api(input_path: str, output_path: str, max_frames: int) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Stable label colors keyed by class name (the API returns names, not ids).
    class_ids: dict[str, int] = {}

    frame_idx = 0
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and frame_idx >= max_frames):
            break
        frame_idx += 1

        for det in detect_via_api(frame):
            cls_id = class_ids.setdefault(det["class"], len(class_ids))
            label = f"{det['class']} {det['confidence']:.2f}"
            draw_box(frame, det["bbox"], label, color_for(cls_id))

        writer.write(frame)
        if total and frame_idx % 25 == 0:
            print(f"  {frame_idx}/{total} frames", end="\r")

    cap.release()
    writer.release()

    wall = time.perf_counter() - t_start
    print(
        f"\nDone: {frame_idx} frames -> {output_path} "
        f"({frame_idx / wall:.1f} FPS end to end through the API)."
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Annotate a video using the /predict API endpoint."
    )
    p.add_argument("input", help="input video path")
    p.add_argument("output", help="output annotated video path (.mp4)")
    p.add_argument(
        "--max-frames", type=int, default=0,
        help="stop after N frames (0 = process the whole video)",
    )
    args = p.parse_args()
    annotate_via_api(args.input, args.output, args.max_frames)


if __name__ == "__main__":
    main()
