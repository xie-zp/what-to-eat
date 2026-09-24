"""Neo4j 知识图谱导入与查询。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import jieba
from neo4j import GraphDatabase

from app.domain.inventory import recipe_matches_inventory
from app.knowledge_graph.models import GraphDataset

NODE_LABELS = (
    "Recipe",
    "Ingredient",
    "CookingStep",
    "Category",
    "IngredientCategory",
    "DifficultyLevel",
)

RELATION_TYPES = (
    "REQUIRES",
    "CONTAINS_STEP",
    "BELONGS_TO_CATEGORY",
    "BELONGS_TO_INGREDIENT_CATEGORY",
    "HAS_DIFFICULTY",
)

QUERY_STOPWORDS = {
    "推荐",
    "请",
    "做法",
    "怎么",
    "如何",
    "一道",
    "几道",
    "菜谱",
    "菜",
    "什么",
    "哪些",
    "哪种",
    "好吃",
    "好",
    "适合",
    "搭配",
    "配",
    "一样",
    "一份",
    "一个",
    "可以",
    "需要",
    "晚餐",
    "早餐",
    "午餐",
}

SEARCH_SYNONYM_GROUPS = {
    "花菜": {
        "花菜",
        "花椰菜",
        "菜花",
    },
    "西兰花": {
        "西兰花",
        "西蓝花",
    },
    "西红柿": {
        "西红柿",
        "番茄",
    },
    "土豆": {
        "土豆",
        "马铃薯",
    },
    "鸡肉": {
        "鸡肉",
        "鸡胸肉",
        "鸡腿",
        "鸡翅",
        "鸡丁",
        "鸡块",
        "鸡丝",
        "鸡胸",
    },
    "蔬菜": {
        "蔬菜",
        "青菜",
        "叶菜",
        "白菜",
        "菠菜",
        "生菜",
        "油麦菜",
        "西兰花",
        "黄瓜",
        "番茄",
        "西红柿",
        "胡萝卜",
        "青椒",
        "菜心",
        "空心菜",
        "蘑菇",
        "香菇",
        "豆芽",
        "韭菜",
        "茄子",
        "西葫芦",
        "土豆",
    },
}


class Neo4jKnowledgeGraph:
    """封装知识图谱的批量导入和基础查询。"""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self) -> None:
        """关闭数据库连接。"""

        self.driver.close()

    def verify_connection(self) -> None:
        """验证数据库连接。"""

        self.driver.verify_connectivity()

    def reset(self) -> None:
        """删除数据库中的所有节点和关系。"""

        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n").consume()

    def create_constraints(self) -> None:
        """为每类节点创建 node_id 唯一约束。"""

        with self.driver.session(database=self.database) as session:
            for label in NODE_LABELS:
                session.run(
                    f"""
                    CREATE CONSTRAINT {label.lower()}_node_id_unique IF NOT EXISTS
                    FOR (n:{label}) REQUIRE n.node_id IS UNIQUE
                    """
                ).consume()

    def import_dataset(self, dataset: GraphDataset) -> dict[str, int]:
        """批量导入节点和关系。"""

        self.create_constraints()
        nodes_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        relationships_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for node in dataset.nodes:
            if node.label not in NODE_LABELS:
                raise ValueError(f"不支持的节点类型：{node.label}")
            nodes_by_label[node.label].append(node.model_dump())

        for relationship in dataset.relationships:
            if relationship.relation_type not in RELATION_TYPES:
                raise ValueError(f"不支持的关系类型：{relationship.relation_type}")
            relationships_by_type[relationship.relation_type].append(relationship.model_dump())

        with self.driver.session(database=self.database) as session:
            for label, rows in nodes_by_label.items():
                for batch in chunked(rows, 500):
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MERGE (n:{label} {{node_id: row.node_id}})
                        SET n.name = row.name
                        SET n += row.properties
                        """,
                        rows=batch,
                    ).consume()

            for relation_type, rows in relationships_by_type.items():
                for batch in chunked(rows, 500):
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MATCH (source {{node_id: row.source_id}})
                        MATCH (target {{node_id: row.target_id}})
                        MERGE (source)-[rel:{relation_type}]->(target)
                        SET rel += row.properties
                        """,
                        rows=batch,
                    ).consume()

        return {
            "nodes": len(dataset.nodes),
            "relationships": len(dataset.relationships),
        }

    def find_recipes(
        self,
        query: str = "",
        category: str | None = None,
        max_difficulty: int | None = None,
        max_minutes: int | None = None,
        exclude_ingredients: Iterable[str] = (),
        available_ingredients: Iterable[str] = (),
        inventory_strict: bool = False,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """根据文本、分类、难度和排除食材查询菜谱。"""

        excluded = [item.strip() for item in exclude_ingredients if item and item.strip()]
        search_terms = build_search_terms(query)
        search_groups = build_search_groups(search_terms)
        has_query = bool(query.strip())
        candidate_limit = 1000 if inventory_strict else max(limit * 5, 20)
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (recipe:Recipe)
                OPTIONAL MATCH (recipe)-[:REQUIRES]->(ingredient:Ingredient)
                WITH recipe, collect(DISTINCT ingredient.name) AS ingredients
                WITH recipe,
                     ingredients,
                     CASE
                         WHEN recipe.name = $search_query THEN 100
                         WHEN recipe.name CONTAINS $search_query THEN 50
                         WHEN $search_query CONTAINS recipe.name THEN 30
                         ELSE 0
                     END AS name_score,
                     reduce(
                         group_score = 0,
                         group IN $search_groups |
                         group_score + CASE
                             WHEN any(
                                 term IN group
                                 WHERE recipe.name CONTAINS term
                                    OR any(
                                        ingredient IN ingredients
                                        WHERE ingredient CONTAINS term
                                    )
                             )
                             THEN 10
                             ELSE 0
                         END
                     ) AS group_score,
                     reduce(
                         description_hits = 0,
                         term IN $search_terms |
                         description_hits + CASE
                             WHEN coalesce(recipe.description, '') CONTAINS term
                             THEN 1
                             ELSE 0
                         END
                     ) AS description_hits
                WITH recipe,
                     ingredients,
                     name_score
                     + group_score
                     + CASE
                         WHEN description_hits > 3 THEN 3
                         ELSE description_hits
                       END AS search_score
                WHERE (NOT $has_query OR search_score > 0)
                  AND ($category IS NULL
                       OR EXISTS {
                           MATCH (recipe)-[:BELONGS_TO_CATEGORY]->(cat:Category)
                           WHERE cat.name = $category
                       })
                  AND ($max_difficulty IS NULL
                       OR recipe.difficulty <= $max_difficulty)
                  AND ($max_minutes IS NULL
                       OR (recipe.estimated_minutes IS NOT NULL
                           AND recipe.estimated_minutes <= $max_minutes))
                  AND all(
                      excluded IN $excluded
                      WHERE none(
                          ingredient IN ingredients
                          WHERE ingredient CONTAINS excluded
                      )
                  )
                OPTIONAL MATCH (recipe)-[:BELONGS_TO_CATEGORY]->(category:Category)
                RETURN recipe.node_id AS recipe_id,
                       recipe.name AS name,
                       recipe.description AS description,
                       recipe.difficulty AS difficulty,
                       recipe.estimated_minutes AS estimated_minutes,
                       ingredients,
                       collect(DISTINCT category.name) AS categories,
                       search_score
                ORDER BY
                    search_score DESC,
                    recipe.difficulty ASC,
                    recipe.name
                LIMIT $limit
                """,
                search_query=query.strip(),
                search_terms=search_terms,
                search_groups=search_groups,
                has_query=has_query,
                category=category,
                max_difficulty=max_difficulty,
                max_minutes=max_minutes,
                excluded=excluded,
                limit=candidate_limit,
            )
            records = [dict(record) for record in result]

        if inventory_strict:
            records = [
                record
                for record in records
                if recipe_matches_inventory(
                    record.get("ingredients") or [],
                    list(available_ingredients),
                )
            ]

        if not records or not has_query:
            return records[:limit]

        max_score = max(float(record.get("search_score") or 0.0) for record in records)
        if max_score >= 10:
            records = [
                record for record in records if float(record.get("search_score") or 0.0) >= 10
            ]
        return records[:limit]

    def get_recipe(self, recipe_id: str) -> dict[str, Any] | None:
        """读取一道完整菜谱。"""

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (recipe:Recipe {node_id: $recipe_id})
                OPTIONAL MATCH (recipe)-[:REQUIRES]->(ingredient:Ingredient)
                OPTIONAL MATCH (recipe)-[step_rel:CONTAINS_STEP]->(step:CookingStep)
                WITH recipe,
                     collect(DISTINCT ingredient.name) AS ingredients,
                     collect(DISTINCT {
                         order: step_rel.order,
                         description: step.description
                     }) AS steps
                RETURN recipe.node_id AS recipe_id,
                       recipe.name AS name,
                       recipe.description AS description,
                       recipe.difficulty AS difficulty,
                       recipe.estimated_minutes AS estimated_minutes,
                       ingredients,
                       steps
                """,
                recipe_id=recipe_id,
            )
            record = result.single()
            if not record:
                return None
            data = dict(record)
            data["steps"] = sorted(
                data.get("steps", []),
                key=lambda item: item.get("order") or 0,
            )
            return data

    def recommend_pairings(
        self,
        anchor_terms: Iterable[str],
        candidate_terms: Iterable[str],
        target_category: str = "",
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """统计与主食材共同出现的候选搭配，并保留菜谱证据。"""

        anchors = [item.strip() for item in anchor_terms if item and item.strip()]
        candidates = [item.strip() for item in candidate_terms if item and item.strip()]
        if not anchors or not candidates:
            return []

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (recipe:Recipe)-[:REQUIRES]->(anchor:Ingredient)
                WHERE any(term IN $anchor_terms WHERE anchor.name CONTAINS term)
                MATCH (recipe)-[:REQUIRES]->(item:Ingredient)
                WHERE none(term IN $anchor_terms WHERE item.name CONTAINS term)
                  AND NOT item.name CONTAINS '、'
                  AND NOT item.name CONTAINS '，'
                  AND item.name <> '蔬菜'
                  AND NOT item.name CONTAINS '蔬菜类'
                  AND NOT item.name CONTAINS '等'
                  AND (
                      ($target_category <> '' AND item.category = $target_category)
                      OR (
                          $target_category = ''
                          AND any(term IN $candidate_terms WHERE item.name CONTAINS term)
                      )
                  )
                WITH item.name AS ingredient,
                     coalesce(item.category, '其他') AS ingredient_category,
                     count(DISTINCT recipe) AS frequency,
                     collect(DISTINCT {
                         recipe_id: recipe.node_id,
                         name: recipe.name,
                         source: 'neo4j:' + recipe.node_id
                     })[0..3] AS examples
                RETURN ingredient, ingredient_category, frequency, examples
                ORDER BY frequency DESC, ingredient
                LIMIT $limit
                """,
                anchor_terms=anchors,
                candidate_terms=candidates,
                target_category=target_category,
                limit=limit,
            )
            return [dict(record) for record in result]


def chunked(values: list[dict[str, Any]], size: int):
    """将列表切分为固定大小的批次。"""

    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_search_terms(query: str) -> list[str]:
    """对中文查询分词，保留具有检索价值的短词。"""

    text = query.strip()
    if not text:
        return []
    terms = [
        token.strip()
        for token in jieba.lcut(text)
        if len(token.strip()) >= 2 and token.strip() not in QUERY_STOPWORDS
    ]
    return list(dict.fromkeys(terms))


def build_search_groups(terms: list[str]) -> list[list[str]]:
    """合并同义概念，供 Cypher 查询按组累计匹配分。"""

    groups: list[list[str]] = []
    for term in terms:
        expanded: set[str] = set()
        for trigger, synonyms in SEARCH_SYNONYM_GROUPS.items():
            if (
                trigger in term
                or term in trigger
                or any(alias in term or term in alias for alias in synonyms)
            ):
                expanded.update(synonyms)
                break
        groups.append(list(dict.fromkeys(expanded or {term})))
    return groups
