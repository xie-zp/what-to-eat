"""LangGraph 状态持久化测试。"""

from concurrent.futures import ThreadPoolExecutor

from app.agent import build_what_to_eat
from app.agent.checkpoint import create_sqlite_checkpointer
from app.state import create_initial_state
from app.tools import RecipeSearchTool, ToolRegistry


def test_agent_state_is_persisted(tmp_path) -> None:
    checkpointer, connection = create_sqlite_checkpointer(tmp_path / "checkpoints.sqlite3")
    registry = ToolRegistry()
    registry.register(RecipeSearchTool())
    graph = build_what_to_eat(
        registry,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "thread-1"}}

    graph.invoke(
        create_initial_state(
            "推荐低糖晚餐",
            {"max_minutes": 30, "allergies": ["花生"]},
        ),
        config=config,
    )
    snapshot = graph.get_state(config)

    assert snapshot.values["status"] == "completed"
    assert "final_answer" in snapshot.values
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    connection.close()


def test_corrupt_checkpoint_is_backed_up_and_recreated(tmp_path) -> None:
    path = tmp_path / "checkpoints.sqlite3"
    path.write_bytes(b"not a sqlite database")

    checkpointer, connection = create_sqlite_checkpointer(path)

    assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert list(tmp_path.glob("checkpoints.sqlite3.corrupt-*"))
    connection.close()


def test_checkpointer_supports_parallel_requests(tmp_path) -> None:
    checkpointer, connection = create_sqlite_checkpointer(tmp_path / "parallel.sqlite3")
    registry = ToolRegistry()
    registry.register(RecipeSearchTool())
    graph = build_what_to_eat(registry, checkpointer=checkpointer)

    def run(index: int) -> str:
        result = graph.invoke(
            create_initial_state("推荐低糖晚餐"),
            config={"configurable": {"thread_id": f"thread-{index}"}},
        )
        return str(result.get("status"))

    with ThreadPoolExecutor(max_workers=4) as executor:
        statuses = list(executor.map(run, range(8)))

    assert statuses == ["completed"] * 8
    assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    connection.close()
