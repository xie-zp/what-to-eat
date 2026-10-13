"""基于 SQLite 的用户画像存储。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """用户长期偏好和硬约束。"""

    user_id: str
    allergies: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    nutrition_goals: list[str] = Field(default_factory=list)
    inventory: list[str] = Field(default_factory=list)


class UserProfileStore:
    """持久化用户画像。"""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )

    def get(self, user_id: str) -> UserProfile:
        """读取用户画像，不存在时返回默认值。"""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            return UserProfile(user_id=user_id)
        return UserProfile.model_validate_json(row["payload"])

    def save(self, profile: UserProfile) -> UserProfile:
        """保存完整用户画像。"""

        payload = profile.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_profiles(user_id, payload)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET payload = excluded.payload
                """,
                (profile.user_id, payload),
            )
        return profile

    def merge(
        self,
        user_id: str,
        updates: dict,
    ) -> UserProfile:
        """合并新的用户画像字段。"""

        current = self.get(user_id).model_dump()
        for key, value in updates.items():
            if key == "user_id":
                continue
            current[key] = value
        return self.save(UserProfile.model_validate(current))

    def to_constraints(self, profile: UserProfile) -> dict:
        """将用户画像转换为 Agent 约束。"""

        return {
            "allergies": profile.allergies,
            "avoid_ingredients": profile.dislikes,
            "inventory": profile.inventory,
            "dietary_goals": profile.nutrition_goals,
        }
