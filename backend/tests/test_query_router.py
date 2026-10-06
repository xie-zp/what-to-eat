"""查询复杂度分析与检索策略路由测试。"""

from app.agent.query_router import SearchStrategy, route_query, strategy_sources


def test_simple_lookup_uses_hybrid_retrieval() -> None:
    for goal in ("宫保鸡丁怎么做？", "川菜有哪些特色菜？", "推荐低糖高蛋白晚餐"):
        analysis = route_query(goal)

        assert analysis.strategy is SearchStrategy.HYBRID_TRADITIONAL, goal


def test_pairing_question_routes_to_graph() -> None:
    analysis = route_query("鸡肉配什么蔬菜好？")

    assert analysis.strategy is SearchStrategy.GRAPH_RAG
    assert analysis.relationship_intensity >= 0.5


def test_multi_condition_inventory_question_routes_to_graph() -> None:
    analysis = route_query("如果我只有鸡肉和蔬菜，能做什么菜，最好是不同菜系的？")

    assert analysis.strategy is SearchStrategy.GRAPH_RAG
    assert analysis.query_complexity > 0.7


def test_substitution_advice_routes_to_graph() -> None:
    analysis = route_query("哪些菜可以用豆腐替代肉类，并且保持相似的口感？")

    assert analysis.strategy is SearchStrategy.GRAPH_RAG
    assert analysis.reasoning_required is True


def test_single_hard_constraint_routes_to_combined() -> None:
    analysis = route_query("冰箱只有花菜，推荐一道菜")

    assert analysis.strategy is SearchStrategy.COMBINED


def test_strategy_sources_mapping() -> None:
    assert strategy_sources("graph_rag") == ["graph"]
    assert strategy_sources("hybrid_traditional") == ["hybrid"]
    assert strategy_sources("combined") == ["hybrid", "graph"]
    assert strategy_sources(None) is None
