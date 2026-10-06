"""Milvus 向量索引与检索。"""

from __future__ import annotations

import logging
import time
from typing import Any

from pymilvus import DataType, MilvusClient

from .documents import RecipeDocument

logger = logging.getLogger(__name__)


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
            for start in range(0, len(documents), batch_size):
                batch = documents[start : start + batch_size]
                vectors = self.embedder.encode_documents([item.text for item in batch])
                rows = [
                    {
                        "id": item.recipe_id,
                        "vector": vector,
                        "text": item.text[:12000],
                        "name": item.name[:300],
                        "category": item.category[:100],
                        "difficulty": item.difficulty,
                        "estimated_minutes": item.estimated_minutes or -1,
                        "ingredients": "、".join(item.ingredients)[:4000],
                        "cuisine_type": item.cuisine_type[:100],
                        "source_path": item.source_path[:1000],
                    }
                    for item, vector in zip(batch, vectors, strict=True)
                ]
                self.client.insert(
                    collection_name=temporary_name,
                    data=rows,
                )
                logger.info(
                    "已写入向量：%s/%s",
                    min(start + batch_size, len(documents)),
                    len(documents),
                )

            self.client.flush(temporary_name)
            self.client.load_collection(temporary_name)
            self._replace_collection(temporary_name)
            return len(documents)
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
            ],
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": 64},
            },
        )
        return [
            {
                "recipe_id": hit["id"],
                "score": hit["distance"],
                "name": hit["entity"]["name"],
                "text": hit["entity"]["text"],
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
