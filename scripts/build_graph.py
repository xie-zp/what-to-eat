"""从标准化菜谱构建知识图谱 JSON。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.domain.recipe import ParsedRecipe
from app.knowledge_graph import build_graph


def load_recipes(path: Path) -> list[ParsedRecipe]:
    """读取解析后的菜谱 JSON。"""

    data = json.loads(path.read_text(encoding="utf-8"))
    return [ParsedRecipe.model_validate(item) for item in data]


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/generated/recipes.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/graph.json"),
    )
    args = parser.parse_args()

    graph = build_graph(load_recipes(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(graph.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已生成 {len(graph.nodes)} 个节点，{len(graph.relationships)} 条关系：{args.output}")


if __name__ == "__main__":
    main()
