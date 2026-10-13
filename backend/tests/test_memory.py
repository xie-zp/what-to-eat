"""用户画像存储测试。"""

from app.memory import UserProfile, UserProfileStore


def test_profile_store_saves_and_merges(tmp_path) -> None:
    store = UserProfileStore(tmp_path / "profiles.sqlite3")
    store.save(
        UserProfile(
            user_id="u1",
            allergies=["花生"],
            inventory=["鸡胸肉"],
        )
    )

    profile = store.merge("u1", {"nutrition_goals": ["低糖"]})

    assert profile.allergies == ["花生"]
    assert profile.inventory == ["鸡胸肉"]
    assert profile.nutrition_goals == ["低糖"]
    assert store.to_constraints(profile)["allergies"] == ["花生"]
