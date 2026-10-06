"""运行 Agent 场景评测并生成报告。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from app.agent import build_recipe_agent
from app.agent.planner import LLMRecipePlanner, RuleBasedPlanner
from app.config import get_settings
from app.state import create_initial_state
from app.tools import create_registry


def load_cases(path: Path) -> list[dict]:
    """读取评测场景。"""

    return json.loads(path.read_text(encoding="utf-8"))


def create_planner(planner_mode: str, registry):
    """按评测模式创建 Planner，缺少模型配置时明确失败。"""

    if planner_mode == "rule":
        return RuleBasedPlanner()
    if planner_mode != "llm":
        raise ValueError(f"不支持的 Planner 模式：{planner_mode}")
    settings = get_settings()
    if not settings.llm_api_key:
        raise RuntimeError("LLM Planner 评测需要配置 LLM_API_KEY")
    return LLMRecipePlanner(
        registry=registry,
        model_name=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


def evaluate(
    cases: list[dict],
    retrieval_mode: str = "neo4j",
    planner_mode: str = "rule",
    fail_on_fallback: bool = False,
) -> dict:
    """运行全部场景并汇总指标。"""

    if retrieval_mode == "hybrid":
        registry = create_registry(use_hybrid=True, use_neo4j=False)
    else:
        registry = create_registry(use_hybrid=False, use_neo4j=True)
    agent = build_recipe_agent(registry, planner=create_planner(planner_mode, registry))
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
        planner_usage = list(output.get("planner_usage") or [])
        if fail_on_fallback:
            ensure_no_planner_fallback(
                case_id=str(case["id"]),
                planner_mode=planner_mode,
                planner_usage=planner_usage,
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
                "planner_usage": planner_usage,
                "citation_coverage": citation_coverage,
                "latency_seconds": latency,
                "evidence": names,
            }
        )

    return {
        "retrieval_mode": retrieval_mode,
        "planner_mode": planner_mode,
        "total": len(results),
        "task_success_rate": average(results, "success"),
        "retrieval_hit_rate": average(results, "retrieval_hit"),
        "relevance_hit_rate": average(results, "relevance_ok"),
        "constraint_satisfaction_rate": average(results, "safety_ok"),
        "citation_coverage": statistics.mean(result["citation_coverage"] for result in results),
        "average_latency_seconds": statistics.mean(result["latency_seconds"] for result in results),
        "planner_calls": sum(len(result["planner_usage"]) for result in results),
        "llm_plan_rate": _planner_rate(results, "llm"),
        "planner_fallback_rate": _planner_rate(results, "llm_fallback"),
        "results": results,
    }


def average(results: list[dict], key: str) -> float:
    """计算布尔指标平均值。"""

    return sum(bool(result[key]) for result in results) / len(results)


def ensure_no_planner_fallback(
    case_id: str,
    planner_mode: str,
    planner_usage: list[str],
) -> None:
    """模型评测出现降级时立即停止，避免把规则结果算作模型结果。"""

    if planner_mode == "llm" and "llm_fallback" in planner_usage:
        raise RuntimeError(f"场景 {case_id} 触发了 LLM Planner 规则降级，本次模型评测无效")


def _planner_rate(results: list[dict], planner_name: str) -> float:
    calls = [item for result in results for item in result.get("planner_usage", [])]
    if not calls:
        return 0.0
    return calls.count(planner_name) / len(calls)


def render_report(metrics: dict) -> str:
    """生成 Markdown 评测报告。"""

    planner_name = {
        "rule": "RuleBasedPlanner",
        "llm": "LLMRecipePlanner（失败时规则降级）",
    }.get(metrics.get("planner_mode"), metrics.get("planner_mode") or "unknown")
    lines = [
        "# Agent 评测报告",
        "",
        f"- 检索模式：{metrics.get('retrieval_mode', 'unknown')}",
        f"- Planner：{planner_name}",
        f"- 场景数量：{metrics['total']}",
        f"- 任务成功率：{metrics['task_success_rate']:.1%}",
        f"- 检索命中率：{metrics['retrieval_hit_rate']:.1%}",
        f"- 相关性命中率：{metrics['relevance_hit_rate']:.1%}",
        f"- 约束满足率：{metrics['constraint_satisfaction_rate']:.1%}",
        f"- 引用覆盖率：{metrics['citation_coverage']:.1%}",
        f"- 平均延迟：{metrics['average_latency_seconds']:.2f} 秒",
    ]
    if metrics.get("planner_mode") == "llm":
        lines.extend(
            [
                f"- Planner 调用次数：{metrics.get('planner_calls', 0)}",
                f"- LLM 直接规划比例：{metrics.get('llm_plan_rate', 0.0):.1%}",
                f"- 规则降级比例：{metrics.get('planner_fallback_rate', 0.0):.1%}",
            ]
        )
    lines.extend(
        [
            "",
            "| 场景 | 成功 | 检索命中 | 相关性 | 安全约束 | 引用覆盖 | 延迟 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
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
        default=None,
    )
    parser.add_argument(
        "--mode",
        choices=("neo4j", "hybrid"),
        default="neo4j",
    )
    parser.add_argument(
        "--planner",
        choices=("rule", "llm"),
        default="rule",
    )
    parser.add_argument(
        "--fail-on-fallback",
        action="store_true",
        help="LLM Planner 出现规则降级时立即失败，避免生成混合评测报告",
    )
    args = parser.parse_args()

    output = args.output
    if output is None:
        output = Path("docs/evaluation-llm.md" if args.planner == "llm" else "docs/evaluation.md")
    metrics = evaluate(
        load_cases(args.cases),
        retrieval_mode=args.mode,
        planner_mode=args.planner,
        fail_on_fallback=args.fail_on_fallback,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(metrics), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
