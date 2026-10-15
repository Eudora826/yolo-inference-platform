"""可选的 OpenTelemetry 链路追踪。

指标告诉你「p95 是 300ms」，链路告诉你「这一次 300ms 里 40ms 解码、
250ms 推理、10ms 编码」。前者用于发现问题，后者用于定位问题，不可
互相替代。

默认关闭：只有环境变量 OTEL_EXPORTER_OTLP_ENDPOINT 有值时才启用，
而 docker-compose.yml 会设置它。不设时 opentelemetry-api 发放的是
non-recording span，所有调用都是空操作 —— 所以镜像可以脱离整套栈
单独 docker run，代价接近零。

模块顶层只 import 了 API 包（很小、纯 Python）；SDK、exporter 和
FastAPI 自动埋点都在 setup() 内部才 import，精简安装也能启动。

⚠️ 版本必须配套。核心包 1.X.0 对应 instrumentation 0.(X+21)b0，
   本项目是 1.27.0 ↔ 0.48b0。混版本的典型症状是 ImportError，
   或者更难查的「span 莫名其妙不上报」。
"""

import logging
from contextlib import contextmanager
from typing import Optional

from opentelemetry import trace

from app import config

logger = logging.getLogger("yolo-service")

# This is the *instrumentation scope*, not the service name — it says which
# code produced a span (visible as otel.library.name on each span), whereas
# service.name in setup() says which process it came from. Two different
# things, so they should not share a string.
_tracer = trace.get_tracer(__name__)
_enabled = False


def setup(app=None) -> bool:
    """装上 OTLP 导出器。返回是否真的启用了追踪。"""
    global _enabled

    endpoint = config.otlp_endpoint()
    if not endpoint:
        logger.info(
            '{"event": "tracing_disabled", '
            '"reason": "OTEL_EXPORTER_OTLP_ENDPOINT not set"}'
        )
        return False

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create({
                "service.name": config.service_name(),
                "service.version": config.service_version(),
            })
        )
        # insecure=True：Collector 在同一个 compose 网络里，不是公网，
        # 没有 TLS 要协商。注意 gRPC 的 endpoint 是 host:port，
        # **不带 http:// 前缀** —— 带了会直接报错。
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)

        if app is not None:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            # /metrics 每 10 秒被 Prometheus 抓一次，/health 被 healthcheck
            # 打得更勤。这些 span 毫无价值，还会把真正的请求淹没。
            FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics")
    except Exception as exc:  # 遥测挂了是小事，服务挂了是大事
        logger.warning('{"event": "tracing_setup_failed", "error": "%s"}', exc)
        return False

    _enabled = True
    logger.info('{"event": "tracing_enabled", "endpoint": "%s"}', endpoint)
    return True


def enabled() -> bool:
    return _enabled


@contextmanager
def span(name: str, **attributes):
    """开一个子 span。追踪关闭时是空操作。

    注意 span 属性和指标标签的基数规则**完全不同**：这里放
    image.width=1280 完全没问题，因为 span 是离散事件不是聚合的
    时间序列；同样的值放进指标标签就是灾难。
    """
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def annotate(**attributes) -> None:
    """给当前活跃的 span 挂属性。"""
    current = trace.get_current_span()
    for key, value in attributes.items():
        if value is not None:
            current.set_attribute(key, value)


def current_trace_id() -> Optional[str]:
    """当前 span 的 trace id（32 位十六进制）。

    写进访问日志，就能从一条报错日志直接跳到 Jaeger 里的完整链路。
    这个动作叫 log-trace correlation，是可观测性成熟度的标志之一。
    """
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x")
