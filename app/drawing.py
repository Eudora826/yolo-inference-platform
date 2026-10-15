"""Drawing helpers shared by the API's annotated endpoints and the demo scripts."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2

# One stable color per class id, so the same object type keeps its color.
_COLOR_CACHE: Dict[int, Tuple[int, int, int]] = {}


def color_for(class_id: int) -> Tuple[int, int, int]:
    if class_id not in _COLOR_CACHE:
        # Deterministic BGR color derived from the class id, so each class
        # keeps a stable color across frames.
        r = (37 * (class_id + 1)) % 256
        g = (17 * (class_id + 3)) % 256
        b = (29 * (class_id + 7)) % 256
        _COLOR_CACHE[class_id] = (int(b), int(g), int(r))
    return _COLOR_CACHE[class_id]


def draw_box(frame, xyxy, label: str, color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # Filled label background above the box for readability.
    cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
    )


def draw_detections(frame, detections: List[dict], class_ids: Dict[str, int]) -> None:
    """Draw a list of API-shaped detections ({class, confidence, bbox}) in place."""
    for det in detections:
        cls_id = class_ids.setdefault(det["class"], len(class_ids))
        label = "{} {:.2f}".format(det["class"], det["confidence"])
        draw_box(frame, det["bbox"], label, color_for(cls_id))
