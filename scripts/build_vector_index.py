"""将菜谱文档写入 Milvus 向量索引。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from app.config import get_settings
from app.rag.documents import build_ingredient_nodes, load_recipe_documents
from app.rag.vector_store import MilvusVectorIndex, SentenceTransformerEmbedder


def main() -> None:
    """命令行入口。"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/generated/recipes.json"),
    )
    args = parser.parse_args()

    settings = get_settings()
    documents = load_recipe_documents(args.input)
    nodes = build_ingredient_nodes(documents)
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        device=os.getenv("EMBEDDING_DEVICE", "cpu"),
    )
    index = MilvusVectorIndex(
        uri=settings.milvus_uri,
        collection_name=settings.milvus_collection,
        dimension=settings.embedding_dimension,
        embedder=embedder,
    )
    try:
        count = index.build(documents, nodes=nodes)
        print(f"Milvus 索引构建完成：{count} 条向量")
    finally:
        index.close()


if __name__ == "__main__":
    main()
