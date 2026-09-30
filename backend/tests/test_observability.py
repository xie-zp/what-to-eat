"""OpenTelemetry 追踪适配层测试。"""

from __future__ import annotations

from app import observability
from app.config import Settings


def test_tracing_is_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(observability, "_tracer", None)
    monkeypatch.setattr(observability, "_tracing_enabled", False)

    with observability.trace_span("test.span", {"value": 1}) as span:
        assert span is None

    assert observability.current_trace_id("fallback") == "fallback"
    assert observability.configure_tracing(Settings(otel_enabled=False)) is False


def test_complex_span_attributes_are_json_encoded() -> None:
    assert observability._json_value({"b": 2, "a": 1}) == '{"b": 2, "a": 1}'
