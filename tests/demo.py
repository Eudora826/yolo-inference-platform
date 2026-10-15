"""
Annotate a video with YOLOv8 detections.

Runs the detector frame by frame, draws bounding boxes and labels, and writes
an annotated MP4. Produces the moving-image artifact used in the portfolio.

Usage (run from the repo root so `app` is importable):
    python -m tests.demo input.mp4 output.mp4
    python -m tests.demo input.mp4 output.mp4 --model yolov8n.pt --conf 0.35
"""

from __future__ import annotations

import argparse
import time

import cv2
from ultralytics import YOLO

from app.drawing import color_for, draw_box


def annotate(input_path: str, output_path: str, model_path: str, conf: float) -> None:
    model = YOLO(model_path)
    names = model.names

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    t_start = time.perf_counter()
    infer_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        t0 = time.perf_counter()
        result = model.predict(frame, conf=conf, verbose=False)[0]
        infer_time += time.perf_counter() - t0

        for box in result.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            label = f"{names[cls_id]} {score:.2f}"
            draw_box(frame, box.xyxy[0].tolist(), label, color_for(cls_id))

        # Overlay a running FPS counter in the corner.
        avg_fps = frame_idx / infer_time if infer_time else 0.0
        cv2.putText(
            frame, f"{avg_fps:5.1f} FPS", (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
        )

        writer.write(frame)
        if total and frame_idx % 25 == 0:
            print(f"  {frame_idx}/{total} frames", end="\r")

    cap.release()
    writer.release()

    wall = time.perf_counter() - t_start
    print(
        f"\nDone: {frame_idx} frames -> {output_path}\n"
        f"Inference {frame_idx / infer_time:.1f} FPS "
        f"(detector only), {frame_idx / wall:.1f} FPS end to end."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Annotate a video with YOLOv8 detections.")
    p.add_argument("input", help="input video path")
    p.add_argument("output", help="output annotated video path (.mp4)")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO weights (default yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    args = p.parse_args()
    annotate(args.input, args.output, args.model, args.conf)


if __name__ == "__main__":
    main()
