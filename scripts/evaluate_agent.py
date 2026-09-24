"""运行 Agent 场景评测并生成报告。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from app.agent import build_recipe_agent
from app.state import create_initial_state
from app.tools import create_registry


def load_cases(path: Path) -> list[dict]:
    """读取评测场景。"""

    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(cases: list[dict], retrieval_mode: str = "neo4j") -> dict:
    """运行全部场景并汇总指标。"""

    if retrieval_mode == "hybrid":
        registry = create_registry(use_hybrid=True, use_neo4j=False)
    else:
        registry = create_registry(use_hybrid=False, use_neo4j=True)
    agent = build_recipe_agent(registry)
    results = []

    for case in cases:
        started = time.perf_counter()
        state = create_initial_state(
            goal=case["goal"],
            constraints=case.get("constraints", {}),
            user_id=f"eval-{case['id']}",
        )
        output = agent.invoke(state)
        latency = time.perf_counter() - started

        evidence = output.get("evidence", [])
        names = [item.name for item in evidence]
        ingredients = [ingredient for item in evidence for ingredient in item.ingredients]
        expected = case.get("expected_recipe")
        retrieval_hit = expected is None or expected in names
        forbidden = case.get("forbidden_ingredients", [])
        safety_ok = not any(
            forbidden_item in ingredient
            for forbidden_item in forbidden
            for ingredient in ingredients
        )
        required_terms = case.get("required_ingredients_any", [])
        relevance_ok = not required_terms or any(
            term in ingredient for term in required_terms for ingredient in ingredients
        )
        meal_plan_ok = not case.get("requires_meal_plan") or bool(output.get("meal_plan"))
        shopping_ok = (
            not case.get("requires_shopping_list") or output.get("shopping_list") is not None
        )
        citation_coverage = (
            sum(bool(item.source) for item in evidence) / len(evidence) if evidence else 0.0
        )
        success = (
            output.get("status") == "completed"
            and retrieval_hit
            and relevance_ok
            and safety_ok
            and meal_plan_ok
            and shopping_ok
        )
        results.append(
            {
                "id": case["id"],
                "success": success,
                "retrieval_hit": retrieval_hit,
                "relevance_ok": relevance_ok,
                "safety_ok": safety_ok,
                "meal_plan_ok": meal_plan_ok,
                "shopping_ok": shopping_ok,
                "citation_coverage": citation_coverage,
                "latency_seconds": latency,
                "evidence": names,
            }
        )

    return {
        "retrieval_mode": retrieval_mode,
        "total": len(results),
        "task_success_rate": average(results, "success"),
        "retrieval_hit_rate": average(results, "retrieval_hit"),
        "relevance_hit_rate": average(results, "relevance_ok"),
        "constraint_satisfaction_rate": average(results, "safety_ok"),
        "citation_coverage": statistics.mean(result["citation_coverage"] for result in results),
        "average_latency_seconds": statistics.mean(result["latency_seconds"] for result in results),
        "results": results,
    }


def average(results: list[dict], key: str) -> float:
    """计算布尔指标平均值。"""

    return sum(bool(result[key]) for result in results) / len(results)


def render_report(metrics: dict) -> str:
    """生成 Markdown 评测报告。"""

    lines = [
        "# Agent 评测报告",
        "",
        f"- 检索模式：{metrics.get('retrieval_mode', 'unknown')}",
        "- Planner：RuleBasedPlanner",
        f"- 场景数量：{metrics['total']}",
        f"- 任务成功率：{metrics['task_success_rate']:.1%}",
        f"- 检索命中率：{metrics['retrieval_hit_rate']:.1%}",
        f"- 相关性命中率：{metrics['relevance_hit_rate']:.1%}",
        f"- 约束满足率：{metrics['constraint_satisfaction_rate']:.1%}",
        f"- 引用覆盖率：{metrics['citation_coverage']:.1%}",
        f"- 平均延迟：{metrics['average_latency_seconds']:.2f} 秒",
        "",
        "| 场景 | 成功 | 检索命中 | 相关性 | 安全约束 | 引用覆盖 | 延迟 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in metrics["results"]:
        lines.append(
            f"| {result['id']} | {result['success']} | "
            f"{result['retrieval_hit']} | {result['relevance_ok']} | "
            f"{result['safety_ok']} | "
            f"{result['citation_coverage']:.0%} | "
            f"{result['latency_seconds']:.2f}s |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evals/cases.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evaluation.md"),
    )
    parser.add_argument(
        "--mode",
        choices=("neo4j", "hybrid"),
        default="neo4j",
    )
    args = parser.parse_args()

    metrics = evaluate(load_cases(args.cases), retrieval_mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
