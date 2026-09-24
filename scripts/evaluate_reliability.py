"""通过真实 HTTP API 运行 Agent 可靠性回归。"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    """读取可靠性场景。"""

    return json.loads(path.read_text(encoding="utf-8"))


def run_case(
    base_url: str,
    case: dict[str, Any],
    timeout: int = 90,
) -> dict[str, Any]:
    """调用真实 Agent API 并返回响应。"""

    body = json.dumps(
        {
            "goal": case["goal"],
            "constraints": case.get("constraints", {}),
            "explicit_constraints": case.get("explicit_constraints", []),
            "history": case.get("history", []),
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/plan",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_case(payload: dict[str, Any], expect: dict[str, Any]) -> list[str]:
    """检查一个接口响应是否满足结构化断言。"""

    failures: list[str] = []
    query_plan = payload.get("query_plan") or {}
    plan_constraints = query_plan.get("constraints") or {}
    evidence = payload.get("evidence") or []
    evidence_names = [str(item.get("name") or "") for item in evidence]
    ingredients = [
        str(ingredient) for item in evidence for ingredient in (item.get("ingredients") or [])
    ]
    tools = [str(item.get("tool_name") or "") for item in payload.get("tool_calls") or []]
    answer = str(payload.get("final_answer") or "")

    if expect.get("intent") and query_plan.get("intent") != expect["intent"]:
        failures.append(f"intent={query_plan.get('intent')!r}，期望 {expect['intent']!r}")
    for key in expect.get("query_plan_constraints_none", []):
        if plan_constraints.get(key) not in (None, [], ""):
            failures.append(
                f"query_plan.constraints.{key} 应为空，实际为 {plan_constraints[key]!r}"
            )
    for key, value in expect.get("query_plan_constraints_equal", {}).items():
        if plan_constraints.get(key) != value:
            failures.append(
                f"query_plan.constraints.{key}={plan_constraints.get(key)!r}，期望 {value!r}"
            )
    for tool_name in expect.get("tools_include", []):
        if tool_name not in tools:
            failures.append(f"缺少工具调用 {tool_name!r}")
    if expect.get("tools_empty") and tools:
        failures.append(f"不应调用工具，实际为 {tools!r}")
    if expect.get("evidence_nonempty") and not evidence:
        failures.append("evidence 不应为空")
    if expect.get("evidence_empty") and evidence:
        failures.append(f"evidence 应为空，实际为 {evidence_names!r}")
    if expect.get("recipe_details_empty") and payload.get("recipe_details"):
        failures.append("recipe_details 应为空")
    if expect.get("blocked_nonempty") and not payload.get("blocked_recipe_ids"):
        failures.append("blocked_recipe_ids 不应为空")
    if expect.get("meal_plan_nonempty") and not payload.get("meal_plan"):
        failures.append("meal_plan 不应为空")
    if expect.get("shopping_list_nonempty") and not payload.get("shopping_list"):
        failures.append("shopping_list 不应为空")

    for value in expect.get("answer_contains", []):
        if value not in answer:
            failures.append(f"回答缺少 {value!r}")
    if expect.get("answer_contains_any") and not any(
        value in answer for value in expect["answer_contains_any"]
    ):
        failures.append(f"回答缺少任一关键词 {expect['answer_contains_any']!r}")
    for value in expect.get("answer_not_contains", []):
        if value in answer:
            failures.append(f"回答不应包含 {value!r}")

    for value in expect.get("evidence_name_contains", []):
        if not any(value in name for name in evidence_names):
            failures.append(f"evidence 缺少名称包含 {value!r} 的菜谱")
    for value in expect.get("evidence_name_not_contains", []):
        if any(value in name for name in evidence_names):
            failures.append(f"evidence 不应包含名称含 {value!r} 的菜谱")

    for value in expect.get("forbidden_ingredients", []):
        if any(value in ingredient for ingredient in ingredients):
            failures.append(f"evidence 含禁用食材 {value!r}")
    for value in expect.get("all_evidence_ingredients_present", []):
        covered = any(
            value in ingredient
            for item in evidence
            for ingredient in (item.get("ingredients") or [])
        )
        if not covered:
            failures.append(f"evidence 未覆盖食材 {value!r}")

    expected_detail = expect.get("recipe_detail_name")
    if expected_detail:
        details = payload.get("recipe_details") or []
        if not any(item.get("name") == expected_detail for item in details):
            failures.append(f"recipe_details 缺少 {expected_detail!r}")

    return failures


def evaluate(
    base_url: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """运行全部可靠性场景并汇总。"""

    results = []
    for case in cases:
        try:
            payload = run_case(base_url, case)
            failures = check_case(payload, case.get("expect", {}))
        except Exception as exc:
            payload = {}
            failures = [f"{type(exc).__name__}: {exc}"]
        results.append(
            {
                "id": case["id"],
                "goal": case["goal"],
                "passed": not failures,
                "failures": failures,
                "intent": (payload.get("query_plan") or {}).get("intent"),
                "tools": [item.get("tool_name") for item in payload.get("tool_calls") or []],
                "evidence": [item.get("name") for item in payload.get("evidence") or []],
                "answer": payload.get("final_answer"),
            }
        )

    passed = sum(item["passed"] for item in results)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "results": results,
    }


def render_report(metrics: dict[str, Any]) -> str:
    """生成 Markdown 报告。"""

    lines = [
        "# Agent 可靠性评测",
        "",
        f"- 场景数量：{metrics['total']}",
        f"- 通过：{metrics['passed']}",
        f"- 失败：{metrics['failed']}",
        f"- 通过率：{metrics['pass_rate']:.1%}",
        "",
        "| 场景 | 结果 | 意图 | 工具 | 失败原因 |",
        "|---|---:|---|---|---|",
    ]
    for item in metrics["results"]:
        failures = "<br>".join(item["failures"]) if item["failures"] else ""
        tools = "、".join(str(tool) for tool in item["tools"])
        lines.append(
            f"| {item['id']} | {'通过' if item['passed'] else '失败'} | "
            f"{item['intent'] or ''} | {tools} | {failures} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evals/reliability_cases.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metrics = evaluate(args.base_url, load_cases(args.cases))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_report(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
