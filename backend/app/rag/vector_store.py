"""Milvus 向量索引与检索。"""

from __future__ import annotations

import logging
import time
from typing import Any

from pymilvus import DataType, MilvusClient

from .documents import RecipeChunk, RecipeDocument

logger = logging.getLogger(__name__)


def safe_truncate(value: Any, max_length: int) -> str:
    """按字段上限截断字符串，避免 Milvus 插入时因超长失败。"""

    if value is None:
        return ""
    return str(value)[:max_length]


def _whole_document_chunk(document: RecipeDocument) -> RecipeChunk:
    """没有预切分时退化为整篇文档入索引。"""

    return RecipeChunk(
        chunk_id=document.recipe_id,
        parent_id=document.recipe_id,
        chunk_type="document",
        text=document.text,
    )


class SentenceTransformerEmbedder:
    """基于 Sentence Transformers 的向量模型封装。"""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        local_files_only: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=local_files_only,
        )

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档。"""

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.tolist()

    def encode_query(self, text: str) -> list[float]:
        """编码查询文本。"""

        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()


class MilvusVectorIndex:
    """管理菜谱向量索引。"""

    def __init__(
        self,
        uri: str,
        collection_name: str,
        dimension: int,
        embedder: Any,
    ) -> None:
        self.uri = uri
        self.collection_name = collection_name
        self.dimension = dimension
        self.embedder = embedder
        self.client = MilvusClient(uri=uri)

    def close(self) -> None:
        """关闭 Milvus 客户端。"""

        self.client.close()

    def build(
        self,
        documents: list[RecipeDocument],
        batch_size: int = 64,
    ) -> int:
        """重建向量集合并写入全部菜谱文档。"""

        temporary_name = f"{self.collection_name}_building_{int(time.time())}"
        if self.client.has_collection(temporary_name):
            self.client.drop_collection(temporary_name)

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=100,
        )
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dimension,
        )
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=12000,
        )
        schema.add_field(
            field_name="name",
            datatype=DataType.VARCHAR,
            max_length=300,
        )
        schema.add_field(
            field_name="category",
            datatype=DataType.VARCHAR,
            max_length=100,
        )
        schema.add_field(
            field_name="difficulty",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="estimated_minutes",
            datatype=DataType.INT64,
        )
        schema.add_field(
            field_name="ingredients",
            datatype=DataType.VARCHAR,
            max_length=4000,
        )
        schema.add_field(
            field_name="cuisine_type",
            datatype=DataType.VARCHAR,
            max_length=100,
        )
        schema.add_field(
            field_name="source_path",
            datatype=DataType.VARCHAR,
            max_length=1000,
        )
        schema.add_field(
            field_name="node_id",
            datatype=DataType.VARCHAR,
            max_length=100,
        )
        schema.add_field(
            field_name="parent_id",
            datatype=DataType.VARCHAR,
            max_length=100,
        )
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            max_length=150,
        )
        schema.add_field(
            field_name="chunk_type",
            datatype=DataType.VARCHAR,
            max_length=50,
        )
        schema.add_field(
            field_name="node_type",
            datatype=DataType.VARCHAR,
            max_length=50,
        )
        schema.add_field(
            field_name="doc_type",
            datatype=DataType.VARCHAR,
            max_length=50,
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self.client.create_collection(
            collection_name=temporary_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )

        try:
            entries = [
                (chunk, document)
                for document in documents
                for chunk in (document.chunks or [_whole_document_chunk(document)])
            ]
            for start in range(0, len(entries), batch_size):
                batch = entries[start : start + batch_size]
                vectors = self.embedder.encode_documents([chunk.text for chunk, _ in batch])
                rows = [
                    {
                        "id": safe_truncate(chunk.chunk_id, 150),
                        "vector": vector,
                        "text": safe_truncate(chunk.text, 12000),
                        "name": safe_truncate(document.name, 300),
                        "category": safe_truncate(document.category, 100),
                        "difficulty": document.difficulty,
                        "estimated_minutes": document.estimated_minutes or -1,
                        "ingredients": safe_truncate(
                            "、".join(document.ingredients),
                            4000,
                        ),
                        "cuisine_type": safe_truncate(document.cuisine_type, 100),
                        "source_path": safe_truncate(document.source_path, 1000),
                        "node_id": safe_truncate(document.recipe_id, 100),
                        "parent_id": safe_truncate(chunk.parent_id, 100),
                        "chunk_id": safe_truncate(chunk.chunk_id, 150),
                        "chunk_type": safe_truncate(chunk.chunk_type, 50),
                        "node_type": "recipe",
                        "doc_type": safe_truncate(document.category, 50),
                    }
                    for (chunk, document), vector in zip(batch, vectors, strict=True)
                ]
                self.client.insert(
                    collection_name=temporary_name,
                    data=rows,
                )
                logger.info(
                    "已写入向量：%s/%s",
                    min(start + batch_size, len(entries)),
                    len(entries),
                )

            self.client.flush(temporary_name)
            self.client.load_collection(temporary_name)
            self._replace_collection(temporary_name)
            return len(entries)
        except Exception:
            if self.client.has_collection(temporary_name):
                self.client.drop_collection(temporary_name)
            raise

    def _replace_collection(self, temporary_name: str) -> None:
        """使用临时集合替换正式集合。"""

        self.client.release_collection(temporary_name)
        if self.client.has_collection(self.collection_name):
            self.client.release_collection(self.collection_name)
            self.client.drop_collection(self.collection_name)
        self.client.rename_collection(temporary_name, self.collection_name)
        self.client.load_collection(self.collection_name)

    def verify(self, expected_count: int | None = None) -> dict[str, Any]:
        """校验集合状态、向量数量和检索链路是否可用。"""

        if not self.client.has_collection(self.collection_name):
            return {"ok": False, "reason": "collection_missing"}

        stats = self.client.get_collection_stats(self.collection_name)
        row_count = int(stats.get("row_count") or 0)
        if row_count <= 0:
            return {"ok": False, "reason": "empty_collection", "row_count": row_count}
        if expected_count is not None and row_count != expected_count:
            return {
                "ok": False,
                "reason": "count_mismatch",
                "row_count": row_count,
                "expected_count": expected_count,
            }

        probe = self.embedder.encode_query("索引自检")
        hits = self.client.search(
            collection_name=self.collection_name,
            data=[probe],
            limit=1,
            output_fields=["name", "text"],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )
        if not hits or not hits[0]:
            return {"ok": False, "reason": "search_returned_nothing", "row_count": row_count}
        return {"ok": True, "row_count": row_count}

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """执行向量相似度检索。"""

        if not self.client.has_collection(self.collection_name):
            raise RuntimeError(f"向量集合不存在：{self.collection_name}")

        vector = self.embedder.encode_query(query)
        results = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            limit=limit,
            output_fields=[
                "text",
                "name",
                "category",
                "difficulty",
                "estimated_minutes",
                "ingredients",
                "cuisine_type",
                "source_path",
                "node_id",
                "parent_id",
                "chunk_id",
                "chunk_type",
            ],
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": 64},
            },
        )
        return [
            {
                "recipe_id": (
                    hit["entity"].get("parent_id") or hit["entity"].get("node_id") or hit["id"]
                ),
                "chunk_id": hit["entity"].get("chunk_id") or hit["id"],
                "chunk_type": hit["entity"].get("chunk_type") or "document",
                "text": hit["entity"]["text"],
                "score": hit["distance"],
                "name": hit["entity"]["name"],
                "category": hit["entity"]["category"],
                "difficulty": hit["entity"]["difficulty"],
                "estimated_minutes": (
                    hit["entity"]["estimated_minutes"]
                    if hit["entity"].get("estimated_minutes", -1) >= 0
                    else None
                ),
                "ingredients": hit["entity"]["ingredients"].split("、"),
                "cuisine_type": hit["entity"]["cuisine_type"],
                "source_path": hit["entity"]["source_path"],
            }
            for hit in results[0]
        ]
