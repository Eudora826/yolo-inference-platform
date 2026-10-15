"""给 demo 栈制造流量。

存在的理由有三个，一个比一个实际：

1. 没有流量的大盘是空的，截图放进 README 是负分。
2. 没有流量就无法验证 PromQL —— rate() 返回空，你不知道查询写对没有。
3. 告警规则没法测试：要验证「错误率超过 5% 会告警」，必须有人真的
   去制造错误请求。

因此这个脚本刻意制造**多样**的流量：多个端点、多种参数、多种图片、
以及一定比例**故意的**坏请求。单一流量会让大盘上一半的面板是死的。

全部通过环境变量配置，见 docker-compose.yml。
"""

from __future__ import annotations

import io
import logging
import os
import random
import signal
import sys
import tempfile
import time

import cv2
import numpy as np
import requests
from PIL import Image, ImageEnhance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("loadgen")

TARGET = os.getenv("TARGET_URL", "http://yolo-service:8000").rstrip("/")
RPS = float(os.getenv("RPS", "3"))
ERROR_RATE = float(os.getenv("ERROR_RATE", "0.06"))
VIDEO_EVERY = float(os.getenv("VIDEO_EVERY_SECONDS", "0"))
SEED_IMAGE = os.getenv("SEED_IMAGE", "cat.jpg")
API_KEY = os.getenv("API_KEY", "")

HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# 收到 SIGTERM 后跑完当前请求就退出。不处理的话 docker compose down
# 要等 10 秒超时才会 SIGKILL 掉它。
_running = True


def _stop(signum, _frame):
    global _running
    _running = False
    log.info("收到信号 %s，正在退出", signum)


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


# ---------------------------------------------------------------------------
# 启动竞态：等服务真的可用
#
# compose 的 healthcheck 已经挡了一层，但健壮的客户端不该因为一次连接
# 失败就死掉 —— 而且这样脚本在 compose 之外也能直接跑。
# ---------------------------------------------------------------------------


def wait_for_service(timeout: float = 300.0) -> bool:
    deadline = time.monotonic() + timeout
    delay = 1.0
    while _running and time.monotonic() < deadline:
        try:
            if requests.get(f"{TARGET}/health", timeout=3).ok:
                log.info("服务已就绪：%s", TARGET)
                return True
        except requests.RequestException as exc:
            log.info("等待 %s（%s）", TARGET, type(exc).__name__)
        time.sleep(delay)
        delay = min(delay * 1.6, 10.0)  # 指数退避，别在启动期把日志刷爆
    return False


# ---------------------------------------------------------------------------
# 图片素材
#
# 一直打同一张图的话，detections_per_image 永远是同一个值、类别分布
# 永远只有一根柱子。这里从一张种子图造出多样性。
# ---------------------------------------------------------------------------

_seed = Image.open(SEED_IMAGE).convert("RGB")


def _jpeg(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _variant_transformed() -> bytes:
    """缩放 / 翻转 / 调亮度。检测结果稳定，但耗时和置信度会有波动。"""
    img = _seed.copy()
    scale = random.uniform(0.4, 1.3)
    img = img.resize((max(64, int(img.width * scale)), max(64, int(img.height * scale))))
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
    return _jpeg(img)


def _variant_composite() -> bytes:
    """把种子图贴 1-5 次到一张画布上，制造不同的检测数量。"""
    width, height = 960, 720
    background = tuple(random.randint(30, 220) for _ in range(3))
    canvas = Image.new("RGB", (width, height), background)
    for _ in range(random.randint(1, 5)):
        scale = random.uniform(0.15, 0.45)
        piece = _seed.resize((
            max(16, int(_seed.width * scale)),
            max(16, int(_seed.height * scale)),
        ))
        canvas.paste(
            piece,
            (random.randint(0, max(1, width - piece.width)),
             random.randint(0, max(1, height - piece.height))),
        )
    return _jpeg(canvas)


def _variant_noise() -> bytes:
    """纯噪声：检测数为 0。让 detections_per_image 的低位桶也有数据。"""
    arr = np.random.randint(
        0, 255, (random.randint(240, 720), random.randint(320, 960), 3), dtype=np.uint8
    )
    return _jpeg(Image.fromarray(arr))


def random_image() -> bytes:
    roll = random.random()
    if roll < 0.65:
        return _variant_transformed()
    if roll < 0.88:
        return _variant_composite()
    return _variant_noise()


def make_video(path: str, frames: int = 60, size=(320, 240), fps: int = 15) -> None:
    """合成一段小视频，省得往仓库里塞几 MB 的二进制素材。"""
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    sprite = _seed.resize((size[0] // 2, size[1] // 2))
    sprite_bgr = cv2.cvtColor(np.array(sprite), cv2.COLOR_RGB2BGR)
    sprite_h, sprite_w = sprite_bgr.shape[:2]
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), 40, dtype=np.uint8)
        x = int((size[0] - sprite_w) * (i / max(1, frames - 1)))
        frame[0:sprite_h, x:x + sprite_w] = sprite_bgr
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# 请求
# ---------------------------------------------------------------------------


def _send(name: str, method: str, path: str, **kwargs) -> None:
    t0 = time.perf_counter()
    try:
        resp = requests.request(
            method, f"{TARGET}{path}", headers=HEADERS, timeout=30, **kwargs
        )
        log.info("%-18s %s %-22s %sms", name, resp.status_code, path,
                 round((time.perf_counter() - t0) * 1000))
    except requests.RequestException as exc:
        log.warning("%-18s ERR %-22s %s", name, path, type(exc).__name__)


def predict(annotated: bool = False) -> None:
    params = {"conf": round(random.uniform(0.15, 0.6), 2)}
    # 有时候加类别过滤，让 /predict 的参数空间不至于单一
    if random.random() < 0.25:
        params["classes"] = random.choice(["cat", "dog", "person", "cat,dog"])
    _send(
        "predict/annotated" if annotated else "predict",
        "POST",
        "/predict/annotated" if annotated else "/predict",
        params=params,
        files={"file": ("frame.jpg", random_image(), "image/jpeg")},
    )


def bad_request() -> None:
    """故意的坏请求 —— 没有它，错误率面板永远是 0，告警也测不了。"""
    kind = random.choice(["not-an-image", "unknown-class", "bad-conf"])
    if kind == "not-an-image":
        # 415
        _send("bad:not-image", "POST", "/predict",
              files={"file": ("notes.txt", b"definitely not an image", "text/plain")})
    elif kind == "unknown-class":
        # 400
        _send("bad:unknown-class", "POST", "/predict",
              params={"classes": "dragon"},
              files={"file": ("frame.jpg", random_image(), "image/jpeg")})
    else:
        # 422，FastAPI 的参数校验
        _send("bad:conf-range", "POST", "/predict",
              params={"conf": 2.0},
              files={"file": ("frame.jpg", random_image(), "image/jpeg")})


def submit_video() -> None:
    path = os.path.join(tempfile.gettempdir(), "loadgen-clip.mp4")
    if not os.path.exists(path):
        make_video(path)
    with open(path, "rb") as fh:
        data = fh.read()
    _send("video:submit", "POST", "/videos",
          params={"conf": 0.3},
          files={"file": ("clip.mp4", data, "video/mp4")})


def one_request() -> None:
    if random.random() < ERROR_RATE:
        return bad_request()
    roll = random.random()
    if roll < 0.60:
        predict()
    elif roll < 0.85:
        predict(annotated=True)
    elif roll < 0.95:
        _send("health", "GET", "/health")
    else:
        _send("model/info", "GET", "/model/info")


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------


def main() -> int:
    log.info("目标 %s | RPS=%s | 错误比例=%s | 视频间隔=%ss",
             TARGET, RPS, ERROR_RATE, VIDEO_EVERY or "关闭")

    if not wait_for_service():
        log.error("服务在超时时间内没有就绪，退出")
        return 1

    interval = 1.0 / RPS if RPS > 0 else 0.5
    # 用"下一次该发的时间点"推进，而不是每次固定 sleep(interval)。
    # 固定 sleep 会漏掉请求本身的耗时，实际速率永远低于配置值。
    next_at = time.monotonic()
    next_video = time.monotonic() + VIDEO_EVERY if VIDEO_EVERY > 0 else float("inf")

    while _running:
        now = time.monotonic()
        if now < next_at:
            # 分片睡眠，这样 SIGTERM 能被及时响应
            time.sleep(min(next_at - now, 0.25))
            continue

        next_at += interval
        if next_at < now:
            # 落后太多（比如服务卡了一会儿）就重置，不要追赶式暴打
            next_at = now + interval

        if now >= next_video:
            submit_video()
            next_video = now + VIDEO_EVERY
            continue

        one_request()

    log.info("已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
