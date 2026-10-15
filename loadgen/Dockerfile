# 构建上下文是仓库根目录（见 docker-compose.yml 的 build.context），
# 这样才能 COPY 到根目录下的 cat.jpg 当种子图。
FROM python:3.11-slim

WORKDIR /app

# opencv-python-headless 不需要 libGL，但仍然链接 libglib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY loadgen/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY loadgen/loadgen.py .
COPY cat.jpg .

RUN useradd -m appuser
USER appuser

# -u 关掉输出缓冲，否则 docker compose logs 要等缓冲区满了才有东西看
CMD ["python", "-u", "loadgen.py"]
