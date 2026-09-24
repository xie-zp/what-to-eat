"""将知识图谱 JSON 导入 Neo4j。"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.knowledge_graph.models import GraphDataset
from app.rag import Neo4jKnowledgeGraph


def load_graph(path: Path) -> GraphDataset:
    """读取图谱 JSON。"""

    return GraphDataset.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/generated/graph.json"),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="导入前清空现有 Neo4j 数据。",
    )
    args = parser.parse_args()

    settings = get_settings()
    graph = Neo4jKnowledgeGraph(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        graph.verify_connection()
        if args.reset:
            graph.reset()
        stats = graph.import_dataset(load_graph(args.input))
        print(f"Neo4j 导入完成：{stats}")
    finally:
        graph.close()


if __name__ == "__main__":
    main()
