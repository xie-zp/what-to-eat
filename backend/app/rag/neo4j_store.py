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
    "Cuisine",
)

RELATION_TYPES = (
    "REQUIRES",
    "CONTAINS_STEP",
    "BELONGS_TO_CATEGORY",
    "BELONGS_TO_INGREDIENT_CATEGORY",
    "HAS_DIFFICULTY",
    "BELONGS_TO_CUISINE",
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
        cuisine: str | None = None,
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
                  AND ($cuisine IS NULL
                       OR EXISTS {
                           MATCH (recipe)-[:BELONGS_TO_CUISINE]->(cuisine:Cuisine)
                           WHERE cuisine.name = $cuisine
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
                       recipe.estimated_calories AS estimated_calories,
                       recipe.estimated_minutes AS estimated_minutes,
                       recipe.nutrition_tags AS nutrition_tags,
                       recipe.cuisineType AS cuisine_type,
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
                cuisine=cuisine,
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
                         description: step.description,
                         methods: step.methods,
                         tools: step.tools,
                         time_estimate: step.time_estimate
                     }) AS steps
                RETURN recipe.node_id AS recipe_id,
                       recipe.name AS name,
                       recipe.description AS description,
                       recipe.difficulty AS difficulty,
                       recipe.estimated_calories AS estimated_calories,
                       recipe.estimated_minutes AS estimated_minutes,
                       recipe.nutrition_tags AS nutrition_tags,
                       recipe.cuisineType AS cuisine_type,
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
        target_keywords: Iterable[str] | None = None,
        limit: int = 8,
        exclude_categories: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """统计与主食材共同出现的候选搭配，并保留菜谱证据。

        target_keywords 可以是任意词：类别（蔬菜/肉类）、具体食材（土豆）、
        甚至菜名片段，只要图谱节点的 name 或 category 命中即算匹配；
        不传时不做目标过滤，返回全部共现食材。
        """

        anchors = [item.strip() for item in anchor_terms if item and item.strip()]
        keywords = [item.strip() for item in (target_keywords or []) if item and item.strip()]
        excluded = [
            item.strip()
            for item in (exclude_categories if exclude_categories is not None else ("调料", "工具"))
            if item and item.strip()
        ]
        if not anchors:
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
                  AND NOT coalesce(item.category, '其他') IN $exclude_categories
                  AND (
                      size($target_keywords) = 0
                      OR any(kw IN $target_keywords WHERE
                                 item.name CONTAINS kw
                                 OR coalesce(item.category, '其他') CONTAINS kw
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
                target_keywords=keywords,
                exclude_categories=excluded,
                limit=limit,
            )
            return [dict(record) for record in result]

    def find_pairing_paths(
        self,
        anchor_terms: Iterable[str],
        target_category: str,
        limit: int = 8,
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        """用变长路径发现主食材与目标类别之间的关联，并返回可解释路径。

        路径形如：鸡肉 -[REQUIRES]- 黄焖鸡 -[REQUIRES]-> 土豆
        -[BELONGS_TO_INGREDIENT_CATEGORY]-> 蔬菜。
        """

        anchors = [item.strip() for item in anchor_terms if item and item.strip()]
        category = target_category.strip()
        if not anchors or not category:
            return []
        # 跳数必须内联到 Cypher，先夹紧到安全范围再拼接。
        depth = max(1, min(int(max_depth), 4))

        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                UNWIND $anchor_terms AS anchor_name
                MATCH (anchor:Ingredient)
                WHERE anchor.name CONTAINS anchor_name

                MATCH path = (anchor)-[*1..{depth}]-(target)
                WHERE NOT anchor = target
                  AND target:IngredientCategory
                  AND target.name = $target_category
                  AND length(path) >= 3

                WITH path, anchor, target,
                     length(path) AS path_len,
                     relationships(path) AS rels,
                     nodes(path) AS path_nodes

                WITH path, anchor, target, path_len, rels, path_nodes,
                     (1.0 / path_len)
                     + (REDUCE(score = 0.0, node IN path_nodes | score + COUNT {{ (node)--() }})
                        / 10.0 / size(path_nodes))
                     + (CASE WHEN ANY(rel IN rels WHERE type(rel) IN $relation_types)
                             THEN 0.3 ELSE 0.0 END) AS relevance

                ORDER BY relevance DESC, path_len ASC
                LIMIT $limit

                RETURN
                    anchor.name AS anchor,
                    target.name AS target,
                    nodes(path)[-2].name AS ingredient,
                    path_len,
                    relevance,
                    [node IN nodes(path) | coalesce(node.name, node.node_id)] AS path_names,
                    [rel IN relationships(path) | type(rel)] AS path_relations,
                    [node IN nodes(path) | labels(node)] AS path_labels
                """,
                anchor_terms=anchors,
                target_category=category,
                relation_types=list(RELATION_TYPES),
                limit=limit,
            )
            return [dict(record) for record in result]

    def expand_from_ingredients(
        self,
        ingredient_groups: Iterable[Iterable[str]],
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """实体级检索：先按键命中食材节点，再取这些节点的共现菜谱。

        这里匹配的是图里的 Ingredient 节点，而不是直接拿文本去匹配菜谱，
        因此“鸡肉”能通过鸡腿、鸡胸肉等具体节点召回候选菜谱。
        每个分组是一个用户词的展开集合，命中分组越多排序越靠前。
        """

        groups = [
            [alias.strip() for alias in group if alias and alias.strip()]
            for group in ingredient_groups
        ]
        groups = [group for group in groups if group]
        if not groups:
            return []

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                UNWIND range(0, size($term_groups) - 1) AS group_index
                WITH group_index, $term_groups[group_index] AS term_group
                UNWIND term_group AS term
                MATCH (ingredient:Ingredient)
                WHERE ingredient.name CONTAINS term
                  AND size(ingredient.name) <= size(term) + 4
                MATCH (recipe:Recipe)-[:REQUIRES]->(ingredient)
                OPTIONAL MATCH (recipe)-[:REQUIRES]->(item:Ingredient)
                WHERE NOT item.name CONTAINS '、'
                  AND NOT item.name CONTAINS '，'
                WITH recipe,
                     sum(
                         CASE
                             WHEN ingredient.name = term THEN 2.0
                             WHEN ingredient.name ENDS WITH term THEN 1.5
                             WHEN ingredient.name STARTS WITH term THEN 1.2
                             ELSE 1.0
                         END
                     ) AS match_score,
                     count(DISTINCT group_index) AS matched_groups,
                     count(DISTINCT ingredient) AS matched_ingredients,
                     collect(DISTINCT item.name)[0..15] AS ingredients
                RETURN recipe.node_id AS recipe_id,
                       recipe.name AS name,
                       recipe.category AS category,
                       recipe.difficulty AS difficulty,
                       recipe.cuisineType AS cuisine_type,
                       matched_groups,
                       matched_ingredients,
                       match_score,
                       ingredients,
                       'neo4j:' + recipe.node_id AS source
                ORDER BY matched_groups DESC, matched_ingredients DESC, match_score DESC,
                         recipe.name
                LIMIT $limit
                """,
                term_groups=groups,
                limit=limit,
            )
            return [dict(record) for record in result]

    def find_by_theme(
        self,
        cuisine: str = "",
        category: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """按菜系或分类节点召回菜谱。

        对应图检索里的主题层：先沿 BELONGS_TO_CUISINE / BELONGS_TO_CATEGORY
        定位到主题节点，再取得属于该主题的菜谱。
        """

        cuisine_name = cuisine.strip()
        category_name = category.strip()
        if not cuisine_name and not category_name:
            return []

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (recipe:Recipe)
                WHERE ($cuisine <> '' AND EXISTS {
                           MATCH (recipe)-[:BELONGS_TO_CUISINE]->(:Cuisine {name: $cuisine})
                       })
                   OR ($category <> '' AND EXISTS {
                           MATCH (recipe)-[:BELONGS_TO_CATEGORY]->(:Category {name: $category})
                       })
                OPTIONAL MATCH (recipe)-[:REQUIRES]->(item:Ingredient)
                WHERE NOT item.name CONTAINS '、'
                  AND NOT item.name CONTAINS '，'
                OPTIONAL MATCH (recipe)-[:BELONGS_TO_CATEGORY]->(category:Category)
                WITH recipe,
                     coalesce(category.name, '未知') AS recipe_category,
                     collect(DISTINCT item.name)[0..15] AS ingredients
                RETURN recipe.node_id AS recipe_id,
                       recipe.name AS name,
                       recipe_category AS category,
                       recipe.difficulty AS difficulty,
                       recipe.cuisineType AS cuisine_type,
                       ingredients,
                       'neo4j:' + recipe.node_id AS source
                ORDER BY recipe.name
                LIMIT $limit
                """,
                cuisine=cuisine_name,
                category=category_name,
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
