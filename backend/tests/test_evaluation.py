"""评测代码的轻量测试。"""

from scripts.evaluate_agent import average, render_report


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
