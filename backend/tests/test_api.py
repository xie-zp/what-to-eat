"""FastAPI 接口契约测试。"""

import app.api.main as api_main
from app.api.main import app
from fastapi.testclient import TestClient


def test_health() -> None:
    response = TestClient(app).get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "search_recipes" in response.json()["tools"]


def test_index_page() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "食序" in response.text


def test_static_assets() -> None:
    client = TestClient(app)
    css = client.get("/static/styles.css")
    script = client.get("/static/app.js")

    assert css.status_code == 200
    assert "--navy" in css.text
    assert script.status_code == 200
    assert "runAgent" in script.text


def test_plan_endpoint() -> None:
    response = TestClient(app).post(
        "/plan",
        json={
            "goal": "推荐低糖晚餐",
            "constraints": {"max_minutes": 30, "allergies": ["花生"]},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["evidence"]
    assert payload["trace"]["total_ms"] >= 0
    assert any(event["name"] == "execute" for event in payload["trace"]["events"])
    assert any(event["name"] == "query_analyzer" for event in payload["trace"]["events"])


def test_profile_roundtrip() -> None:
    client = TestClient(app)
    response = client.put(
        "/profiles/test-user",
        json={
            "allergies": ["花生"],
            "inventory": ["鸡胸肉"],
            "household_size": 2,
        },
    )
    assert response.status_code == 200

    profile = client.get("/profiles/test-user").json()
    assert profile["allergies"] == ["花生"]
    assert profile["inventory"] == ["鸡胸肉"]
    assert profile["household_size"] == 2


def test_plan_can_pause_and_resume_after_confirmation() -> None:
    client = TestClient(app)
    response = client.post(
        "/plan",
        json={
            "goal": "规划两天晚餐菜单并生成购物清单",
            "constraints": {"days": 2, "meals_per_day": 1},
            "confirmation_required": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "awaiting_confirmation"
    assert payload["pending_tool"] == "create_meal_plan"

    thread_id = payload["thread_id"]
    confirmation = client.post(
        f"/tasks/{thread_id}/confirm",
        json={"approved": True, "tools": ["create_meal_plan"]},
    )
    assert confirmation.status_code == 200
    assert confirmation.json()["pending_tool"] == "create_shopping_list"

    confirmation = client.post(
        f"/tasks/{thread_id}/confirm",
        json={"approved": True, "tools": ["create_shopping_list"]},
    )
    assert confirmation.status_code == 200
    assert confirmation.json()["status"] == "completed"


def test_pairing_question_returns_vegetables_directly() -> None:
    response = TestClient(app).post(
        "/plan",
        json={"goal": "鸡肉配什么蔬菜好？"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["pairing_recommendations"]
    assert "适合搭配这些蔬菜" in payload["final_answer"]


def test_unknown_recipe_does_not_fall_back_to_another_recipe() -> None:
    response = TestClient(app).post(
        "/plan",
        json={"goal": "佛跳墙怎么做"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "没有找到" in payload["final_answer"]
    assert payload["recipe_details"] == []


def test_follow_up_how_to_uses_previous_recipe_context() -> None:
    response = TestClient(app).post(
        "/plan",
        json={
            "goal": "怎么做",
            "history": [
                {
                    "role": "assistant",
                    "content": "为你找到以下候选菜谱：\n- 青椒鸡胸肉：20 分钟",
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["recipe_details"][0]["name"] == "青椒鸡胸肉"


def test_blank_goal_returns_structured_validation_error() -> None:
    response = TestClient(app).post("/plan", json={"goal": "   "})

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"] == "请求参数不合法"
    assert payload["error_type"] == "validation_error"


def test_unhandled_error_returns_structured_json(monkeypatch) -> None:
    def fail_invoke(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(api_main.agent_graph, "invoke", fail_invoke)
    response = TestClient(app, raise_server_exceptions=False).post(
        "/plan",
        json={"goal": "推荐一道菜"},
    )

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"] == "服务执行失败，请稍后重试。"
    assert payload["error_type"] == "RuntimeError"


def test_blocked_recipe_is_not_returned_as_evidence(monkeypatch) -> None:
    class StaticAnalyzer:
        def analyze(self, goal, history=None):
            return {
                "intent": "recommendation",
                "entities": ["花生", "宫保鸡丁"],
                "target_entities": [],
                "constraints": {"allergies": ["花生"]},
            }

    monkeypatch.setattr(api_main, "query_analyzer", StaticAnalyzer())
    response = TestClient(app).post(
        "/plan",
        json={"goal": "我对花生过敏，推荐宫保鸡丁"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "r003" in payload["blocked_recipe_ids"]
    evidence_names = [item["name"] for item in payload["evidence"]]
    assert "宫保鸡丁" not in evidence_names
    assert len(evidence_names) == len(set(evidence_names))


def test_stale_form_constraints_are_not_applied_to_new_topic(monkeypatch) -> None:
    class StaticAnalyzer:
        def analyze(self, goal, history=None):
            return {
                "intent": "recommendation",
                "entities": [],
                "target_entities": [],
                "constraints": {"cuisine": "川菜"},
            }

    monkeypatch.setattr(api_main, "query_analyzer", StaticAnalyzer())
    response = TestClient(app).post(
        "/plan",
        json={
            "goal": "川菜有哪些特色菜？",
            "constraints": {
                "max_minutes": 30,
                "inventory": ["鸡蛋"],
                "allergies": ["花生"],
            },
            "explicit_constraints": [],
        },
    )

    assert response.status_code == 200
    constraints = response.json()["query_plan"]["constraints"]
    assert constraints["cuisine"] == "川菜"
    assert constraints["allergies"] == ["花生"]
    assert constraints.get("max_minutes") is None
    assert constraints.get("inventory", []) == []
