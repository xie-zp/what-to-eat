"""可选的 OpenTelemetry 全链路追踪。"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
except ImportError:  # pragma: no cover - 未安装可观测性依赖时走无操作路径
    trace = None
    SpanKind = None
    Status = None
    StatusCode = None

_tracer: Any | None = None
_tracing_enabled = False


def configure_tracing(settings: Settings) -> bool:
    """按配置初始化全局 TracerProvider，重复调用不会创建多个 Provider。"""

    global _tracer, _tracing_enabled
    if not settings.otel_enabled or _tracing_enabled:
        return _tracing_enabled
    if trace is None:
        logger.warning("启用 OpenTelemetry 失败：未安装 observability 可选依赖")
        return False

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": settings.otel_service_name,
                    "service.version": "0.1.0",
                    "deployment.environment": settings.otel_environment,
                }
            )
        )
        exporter = _create_exporter(
            settings.otel_exporter,
            settings.otel_exporter_endpoint,
            ConsoleSpanExporter,
        )
        if exporter is None:
            logger.warning("OpenTelemetry 已启用，但没有可用的 trace exporter")
            return False
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("what-to-eat")
        _tracing_enabled = True
        logger.info(
            "OpenTelemetry 追踪已启用：exporter=%s service=%s",
            settings.otel_exporter,
            settings.otel_service_name,
        )
        return True
    except Exception:
        logger.exception("初始化 OpenTelemetry 失败")
        return False


def _create_exporter(
    exporter_name: str,
    endpoint: str,
    console_exporter_class: Any,
) -> Any | None:
    """创建标准输出或 OTLP HTTP exporter。"""

    normalized = exporter_name.strip().lower()
    if normalized == "console":
        return console_exporter_class()
    if normalized == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter(endpoint=endpoint)
    return None


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: str = "internal",
) -> Iterator[Any | None]:
    """创建 span；追踪关闭或依赖缺失时自动退化为无操作上下文。"""

    if _tracer is None:
        yield None
        return

    span_kind = _span_kind(kind)
    with _tracer.start_as_current_span(name, kind=span_kind) as span:
        set_span_attributes(span, attributes or {})
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def set_span_attributes(span: Any | None, attributes: dict[str, Any]) -> None:
    """写入兼容 OpenTelemetry 的标量属性，复杂值转为 JSON。"""

    if span is None:
        return
    for key, value in attributes.items():
        if value is None or value == "":
            continue
        if isinstance(value, (bool, int, float, str)):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, _json_value(value))


def record_llm_response(span: Any | None, response: Any) -> None:
    """记录 LangChain/OpenAI 兼容响应中的 token 用量。"""

    if span is None:
        return
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        return
    set_span_attributes(
        span,
        {
            "gen_ai.usage.input_tokens": usage.get("input_tokens"),
            "gen_ai.usage.output_tokens": usage.get("output_tokens"),
            "gen_ai.usage.total_tokens": usage.get("total_tokens"),
        },
    )


def trace_node(name: str, attribute_builder=None):
    """为 LangGraph 节点统一创建 span，并自动记录结果状态。"""

    def decorator(func):
        @wraps(func)
        def wrapped(state, *args, **kwargs):
            attributes = attribute_builder(state) if attribute_builder else {}
            with trace_span(name, attributes) as span:
                result = func(state, *args, **kwargs)
                if isinstance(result, dict):
                    set_span_attributes(span, _node_result_attributes(result))
                return result

        return wrapped

    return decorator


def current_trace_id(default: str = "") -> str:
    """返回当前有效 span 的 32 位 trace id。"""

    if trace is None:
        return default
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return default
    return format(context.trace_id, "032x")


def tracing_enabled() -> bool:
    """返回当前进程是否已启用追踪。"""

    return _tracing_enabled


def _span_kind(value: str) -> Any:
    if SpanKind is None:
        return None
    normalized = value.strip().lower()
    if normalized == "server":
        return SpanKind.SERVER
    if normalized == "client":
        return SpanKind.CLIENT
    return SpanKind.INTERNAL


def _json_value(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _node_result_attributes(result: dict[str, Any]) -> dict[str, Any]:
    current_task = result.get("current_task")
    tool_name = None
    if isinstance(current_task, dict):
        tool_name = current_task.get("tool_name")
    return {
        "agent.result.status": result.get("status"),
        "agent.result.tool_name": tool_name,
        "agent.result.error_count": len(result.get("validation_errors") or []),
    }
