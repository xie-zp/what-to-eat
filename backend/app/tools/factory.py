"""工具注册表工厂。"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import get_settings
from app.rag import Neo4jKnowledgeGraph
from app.rag.documents import load_recipe_documents
from app.rag.hybrid import HybridRecipeSearcher
from app.rag.vector_store import MilvusVectorIndex, SentenceTransformerEmbedder

from .inventory import InventoryTool
from .meal_plan import MealPlanTool
from .neo4j_recipe_search import Neo4jRecipeSearchTool
from .nutrition import NutritionTool
from .pairing_recommendation import PairingRecommendationTool
from .recipe_detail import (
    FileRecipeDetailTool,
    Neo4jRecipeDetailTool,
    RecipeDetailTool,
)
from .recipe_search import RecipeSearchTool
from .registry import ToolRegistry
from .revise_meal_plan import ReviseMealPlanTool
from .revise_shopping_list import ReviseShoppingListTool
from .shopping_list import ShoppingListTool
from .unified_recipe_search import UnifiedRecipeSearchTool

logger = logging.getLogger(__name__)


def create_registry(
    use_neo4j: bool | None = None,
    use_hybrid: bool | None = None,
) -> ToolRegistry:
    """创建默认工具注册表。"""

    settings = get_settings()
    use_neo4j = settings.use_neo4j if use_neo4j is None else use_neo4j
    use_hybrid = settings.use_hybrid if use_hybrid is None else use_hybrid
    registry = ToolRegistry()
    if use_hybrid:
        try:
            documents = load_recipe_documents(Path(settings.recipe_data_path))
            embedder = SentenceTransformerEmbedder(
                settings.embedding_model,
                local_files_only=settings.embedding_local_files_only,
            )
            vector_index = MilvusVectorIndex(
                uri=settings.milvus_uri,
                collection_name=settings.milvus_collection,
                dimension=settings.embedding_dimension,
                embedder=embedder,
            )
            searcher = HybridRecipeSearcher(documents, vector_index)
            graph = None
            if use_neo4j:
                graph = Neo4jKnowledgeGraph(
                    uri=settings.neo4j_uri,
                    user=settings.neo4j_user,
                    password=settings.neo4j_password,
                    database=settings.neo4j_database,
                )
            registry.register(UnifiedRecipeSearchTool(searcher=searcher, graph=graph))
            registry.register(FileRecipeDetailTool(settings.recipe_data_path))
            registry.register(PairingRecommendationTool(graph))
        except Exception as exc:
            logger.warning("Hybrid 初始化失败，降级到图谱/内存检索：%s", exc)
            use_hybrid = False
    if not use_hybrid and use_neo4j:
        graph = Neo4jKnowledgeGraph(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        registry.register(Neo4jRecipeSearchTool(graph))
        registry.register(Neo4jRecipeDetailTool(graph))
        registry.register(PairingRecommendationTool(graph))
    elif not use_hybrid:
        registry.register(RecipeSearchTool())
        registry.register(RecipeDetailTool())
        registry.register(PairingRecommendationTool())
    registry.register(NutritionTool())
    registry.register(InventoryTool())
    registry.register(MealPlanTool())
    registry.register(ShoppingListTool())
    registry.register(ReviseMealPlanTool())
    registry.register(ReviseShoppingListTool())
    return registry
