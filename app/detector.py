import time
from typing import Dict, List, Optional, Tuple

from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        # yolov8n is the smallest model; weights download on first use.
        self.model_path = model_path
        self.model = YOLO(model_path)

    @property
    def names(self) -> Dict[int, str]:
        return self.model.names

    def class_ids_for(self, class_names: List[str]) -> List[int]:
        """Map class names to model class ids. Raises ValueError on unknown names."""
        by_name = {name: cls_id for cls_id, name in self.names.items()}
        unknown = [n for n in class_names if n not in by_name]
        if unknown:
            raise ValueError(
                "unknown class name(s): {}".format(", ".join(sorted(unknown)))
            )
        return [by_name[n] for n in class_names]

    def predict(
        self,
        image,  # PIL.Image or BGR numpy array, both accepted by ultralytics
        conf: float = 0.25,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        max_det: int = 100,
    ) -> Tuple[List[dict], float]:
        """Run detection. Returns (detections, inference_time_ms)."""
        t0 = time.perf_counter()
        results = self.model.predict(
            image, conf=conf, iou=iou, classes=classes, max_det=max_det,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": self.names[int(box.cls)],
                    "confidence": round(float(box.conf), 4),
                    "bbox": [round(float(x), 2) for x in box.xyxy[0]],
                })
        return detections, elapsed_ms
