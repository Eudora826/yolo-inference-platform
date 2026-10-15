"""Pydantic response models. These drive the OpenAPI docs at /docs.

Note: typing.List/Dict/Optional (not list[...]) so the service also runs
on Python 3.8, where builtin generics cannot be evaluated by Pydantic.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class Detection(BaseModel):
    class_name: str = Field(alias="class", examples=["person"])
    confidence: float = Field(examples=[0.93])
    bbox: List[float] = Field(
        description="Pixel coordinates [x1, y1, x2, y2]",
        examples=[[102.5, 44.0, 380.2, 512.9]],
    )

    model_config = {"populate_by_name": True}


class ImageSize(BaseModel):
    width: int
    height: int


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    request_id: str
    model_version: str = Field(examples=["yolov8n.pt"])
    inference_time_ms: float
    image_size: ImageSize
    count: int
    detections: List[Detection]


class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_name: str
    task: str = Field(examples=["detect"])
    num_classes: int
    class_names: Dict[int, str]
    ultralytics_version: str
    device: str = Field(examples=["cpu"])


class JobInfo(BaseModel):
    job_id: str
    status: str = Field(description="queued | processing | completed | failed")
    progress: float = Field(description="0.0 to 1.0")
    frames_done: int
    frames_total: int
    error: Optional[str] = None
    result_url: Optional[str] = Field(
        default=None, description="Download URL once the job is completed"
    )


class JobCreated(BaseModel):
    job_id: str
    status: str
    status_url: str
