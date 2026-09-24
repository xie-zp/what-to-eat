"""应用配置。"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Settings(BaseModel):
    """从环境变量加载的运行时配置。"""

    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    neo4j_uri: str = "bolt://localhost:7688"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "all-in-rag"
    neo4j_database: str = "neo4j"
    milvus_uri: str = "http://localhost:19531"
    milvus_collection: str = "recipe_knowledge"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimension: int = 512
    embedding_local_files_only: bool = True
    agent_db_path: str = "./data/agent_state.sqlite3"
    checkpoint_db_path: str = "./data/checkpoints.sqlite3"
    recipe_data_path: str = "./data/generated/recipes.json"
    max_agent_steps: int = 12
    use_neo4j: bool = False
    use_hybrid: bool = False
    use_llm_planner: bool = False
    use_llm_query_analyzer: bool = False
    use_llm_responder: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存的应用程序配置。"""

    return Settings(
        llm_api_key=(os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""),
        llm_base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        llm_model=os.getenv("LLM_MODEL", "deepseek-v4-flash"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7688"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "all-in-rag"),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19531"),
        milvus_collection=os.getenv(
            "MILVUS_COLLECTION",
            "recipe_knowledge",
        ),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "BAAI/bge-small-zh-v1.5",
        ),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "512")),
        embedding_local_files_only=os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "true").lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        },
        agent_db_path=os.getenv("AGENT_DB_PATH", "./data/agent_state.sqlite3"),
        checkpoint_db_path=os.getenv(
            "CHECKPOINT_DB_PATH",
            "./data/checkpoints.sqlite3",
        ),
        recipe_data_path=os.getenv(
            "RECIPE_DATA_PATH",
            "./data/generated/recipes.json",
        ),
        max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "12")),
        use_neo4j=os.getenv("USE_NEO4J", "false").lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        },
        use_hybrid=os.getenv("USE_HYBRID", "false").lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        },
        use_llm_planner=os.getenv("USE_LLM_PLANNER", "false").lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        },
        use_llm_query_analyzer=os.getenv("USE_LLM_QUERY_ANALYZER", "false").lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        },
        use_llm_responder=os.getenv("USE_LLM_RESPONDER", "false").lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        },
    )
