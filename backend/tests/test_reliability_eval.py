"""真实接口可靠性评测工具测试。"""

from scripts.evaluate_reliability import check_case, render_report


def test_check_case_detects_structured_failures() -> None:
    payload = {
        "query_plan": {"intent": "recommendation"},
        "tool_calls": [{"tool_name": "search_recipes"}],
        "evidence": [
            {
                "name": "糖拌西红柿",
                "ingredients": ["西红柿", "花生"],
            }
        ],
        "final_answer": "没有找到",
        "blocked_recipe_ids": [],
        "meal_plan": None,
        "shopping_list": [],
        "recipe_details": [],
    }
    failures = check_case(
        payload,
        {
            "intent": "recommendation",
            "tools_include": ["search_recipes"],
            "evidence_name_contains": ["花菜"],
            "evidence_name_not_contains": ["糖拌"],
            "forbidden_ingredients": ["花生"],
            "blocked_nonempty": True,
        },
    )

    assert len(failures) == 4
    assert any("缺少名称包含" in failure for failure in failures)
    assert any("不应包含" in failure for failure in failures)
    assert any("禁用食材" in failure for failure in failures)
    assert any("blocked_recipe_ids" in failure for failure in failures)


def test_reliability_report_contains_summary() -> None:
    report = render_report(
        {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "pass_rate": 1.0,
            "results": [
                {
                    "id": "case-1",
                    "goal": "测试",
                    "passed": True,
                    "failures": [],
                    "intent": "recommendation",
                    "tools": ["search_recipes"],
                    "evidence": [],
                    "answer": "ok",
                }
            ],
        }
    )

    assert "Agent 可靠性评测" in report
    assert "case-1" in report


def test_check_case_validates_menu_ingredients_and_time() -> None:
    payload = {
        "query_plan": {"intent": "meal_plan"},
        "tool_calls": [],
        "evidence": [
            {
                "name": "超时菜",
                "ingredients": ["鸡肉"],
                "estimated_minutes": 60,
            }
        ],
        "meal_plan": {
            "days": [
                {
                    "day": 1,
                    "meals": [
                        {
                            "name": "花生菜",
                            "ingredients": ["花生"],
                        }
                    ],
                }
            ]
        },
        "final_answer": "ok",
        "blocked_recipe_ids": [],
        "shopping_list": [],
        "recipe_details": [],
    }

    failures = check_case(
        payload,
        {
            "max_evidence_count": 0,
            "forbidden_ingredients_anywhere": ["花生"],
            "all_evidence_minutes_lte": [30],
        },
    )

    assert len(failures) == 3
    assert any("数量" in failure for failure in failures)
    assert any("菜单或证据" in failure for failure in failures)
    assert any("超过 30 分钟" in failure for failure in failures)
