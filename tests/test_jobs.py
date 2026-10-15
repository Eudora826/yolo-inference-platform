import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture()
def tiny_video(tmp_path, monkeypatch):
    """A 6-frame 64x64 video; jobs write under the test's tmp dir."""
    monkeypatch.setenv("JOBS_DIR", str(tmp_path / "jobs"))
    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64)
    )
    for i in range(6):
        frame = np.full((64, 64, 3), i * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def submit(path):
    with open(path, "rb") as f:
        return client.post(
            "/videos", files={"file": ("tiny.mp4", f.read(), "video/mp4")}
        )


def test_video_job_lifecycle(tiny_video):
    # TestClient runs background tasks synchronously, so the job is done
    # by the time the response returns.
    response = submit(tiny_video)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert response.json()["status_url"] == "/jobs/{}".format(job_id)

    status = client.get("/jobs/{}".format(job_id)).json()
    assert status["status"] == "completed"
    assert status["progress"] == 1.0
    assert status["frames_done"] == 6
    assert status["result_url"] == "/jobs/{}/result".format(job_id)

    result = client.get("/jobs/{}/result".format(job_id))
    assert result.status_code == 200
    assert result.headers["content-type"] == "video/mp4"
    assert len(result.content) > 0


def test_unknown_job_returns_404():
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/jobs/nope/result").status_code == 404


def test_wrong_video_format_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_DIR", str(tmp_path / "jobs"))
    response = client.post(
        "/videos", files={"file": ("clip.gif", b"GIF89a", "image/gif")}
    )
    assert response.status_code == 415
