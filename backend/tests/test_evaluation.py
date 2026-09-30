"""评测代码的轻量测试。"""

import pytest
from app.config import Settings

from scripts.evaluate_agent import _planner_rate, average, create_planner, render_report


def test_average_bool_metric() -> None:
    assert average([{"ok": True}, {"ok": False}], "ok") == 0.5


def test_render_report_contains_metrics() -> None:
    report = render_report(
        {
            "total": 1,
            "task_success_rate": 1.0,
            "retrieval_hit_rate": 1.0,
            "relevance_hit_rate": 1.0,
            "constraint_satisfaction_rate": 1.0,
            "citation_coverage": 1.0,
            "average_latency_seconds": 0.5,
            "results": [
                {
                    "id": "case-1",
                    "success": True,
                    "retrieval_hit": True,
                    "relevance_ok": True,
                    "safety_ok": True,
                    "citation_coverage": 1.0,
                    "latency_seconds": 0.5,
                }
            ],
        }
    )

    assert "任务成功率" in report
    assert "case-1" in report


def test_planner_rate_counts_llm_and_fallback_calls() -> None:
    results = [
        {"planner_usage": ["llm", "llm"]},
        {"planner_usage": ["llm_fallback"]},
    ]

    assert _planner_rate(results, "llm") == pytest.approx(2 / 3)
    assert _planner_rate(results, "llm_fallback") == pytest.approx(1 / 3)


def test_llm_report_includes_planner_statistics() -> None:
    report = render_report(
        {
            "retrieval_mode": "neo4j",
            "planner_mode": "llm",
            "planner_calls": 3,
            "llm_plan_rate": 2 / 3,
            "planner_fallback_rate": 1 / 3,
            "total": 1,
            "task_success_rate": 1.0,
            "retrieval_hit_rate": 1.0,
            "relevance_hit_rate": 1.0,
            "constraint_satisfaction_rate": 1.0,
            "citation_coverage": 1.0,
            "average_latency_seconds": 0.5,
            "results": [
                {
                    "id": "case-1",
                    "success": True,
                    "retrieval_hit": True,
                    "relevance_ok": True,
                    "safety_ok": True,
                    "citation_coverage": 1.0,
                    "latency_seconds": 0.5,
                }
            ],
        }
    )

    assert "LLMRecipePlanner" in report
    assert "LLM 直接规划比例：66.7%" in report
    assert "规则降级比例：33.3%" in report


def test_llm_planner_evaluation_requires_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.evaluate_agent.get_settings",
        lambda: Settings(llm_api_key=""),
    )

    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        create_planner("llm", registry=None)
