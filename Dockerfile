# ---- Stage 1: builder ----
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 权重在构建时拉取，不进 git —— 6.3 MB 的二进制一旦提交就永久留在
# 历史里，而 Ultralytics 已经把它托管在 Releases 上了。
#
# 在这里下载（而不是 COPY 一个本地文件）是为了让 clone 之后直接
# `docker compose up --build` 就能跑。早先的版本 COPY 一个被
# .gitignore 挡住的文件，于是任何人照 README 跑第一条命令就会失败 ——
# 而 CI 因为每个 job 都先 wget 补上了文件，完全看不出这个问题。
#
# ADD 原生支持远程 URL，不用再装 curl 或 wget。
ARG WEIGHTS_URL=https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
ADD ${WEIGHTS_URL} /weights/yolov8n.pt

# ---- Stage 2: runtime ----
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd -m appuser

COPY --chown=appuser:appuser app/ ./app/

COPY --from=builder --chown=appuser:appuser /weights/yolov8n.pt ./yolov8n.pt

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
