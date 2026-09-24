"""LangGraph SQLite Checkpointer 工具。"""

from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger(__name__)


def create_sqlite_checkpointer(
    path: str | Path,
) -> tuple[SqliteSaver, sqlite3.Connection]:
    """创建持久化 Checkpointer 和底层连接。"""

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = _connect(db_path)
    if not _database_is_healthy(connection):
        connection.close()
        backup_path = _move_corrupt_database(db_path)
        logger.warning("检测到损坏的 Checkpointer，已备份到 %s 并重建", backup_path)
        connection = _connect(db_path)
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("app.state", "RecipeEvidence"),
            ("app.state", "ToolCallRecord"),
            ("app.state", "ToolResult"),
            ("app.state", "TraceEvent"),
        ]
    )
    return SqliteSaver(connection, serde=serde), connection


def _connect(path: Path) -> sqlite3.Connection:
    """建立 Checkpointer 数据库连接。"""

    return sqlite3.connect(
        path,
        check_same_thread=False,
    )


def _database_is_healthy(connection: sqlite3.Connection) -> bool:
    """使用 SQLite quick_check 判断数据库是否可继续使用。"""

    try:
        row = connection.execute("PRAGMA quick_check").fetchone()
    except sqlite3.DatabaseError:
        return False
    return bool(row and row[0] == "ok")


def _move_corrupt_database(path: Path) -> Path:
    """保留损坏数据库及其 WAL 文件，方便后续排查。"""

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.name}.corrupt-{timestamp}")
    for suffix in ("", "-wal", "-shm"):
        source = Path(f"{path}{suffix}")
        if source.exists():
            source.replace(Path(f"{backup_path}{suffix}"))
    return backup_path
