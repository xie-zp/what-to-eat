"""食谱规划 Agent 的 FastAPI 入口。"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.agent import build_what_to_eat
from app.agent.checkpoint import create_sqlite_checkpointer
from app.agent.constraint_scope import build_effective_constraints
from app.agent.planner import LLMRecipePlanner
from app.agent.query_analyzer import create_query_analyzer
from app.agent.responder import create_responder
from app.config import get_settings
from app.memory import UserProfile, UserProfileStore
from app.observability import (
    configure_tracing,
    current_trace_id,
    set_span_attributes,
    trace_span,
)
from app.state import create_initial_state
from app.tools import create_registry

logger = logging.getLogger(__name__)


class PlanRequest(BaseModel):
    """食谱规划任务请求体。"""

    goal: str = Field(min_length=1, max_length=500)
    constraints: dict = Field(default_factory=dict)
    explicit_constraints: list[str] = Field(default_factory=list)
    user_id: str = "local-user"
    thread_id: str | None = None
    conversation_id: str = ""
    history: list[dict] = Field(default_factory=list)
    meal_plan: dict | None = None
    shopping_list: list[dict] = Field(default_factory=list)
    confirmation_required: bool = False

    @field_validator("goal")
    @classmethod
    def goal_must_not_be_blank(cls, value: str) -> str:
        """拒绝只包含空白的任务描述。"""

        value = value.strip()
        if not value:
            raise ValueError("goal 不能为空")
        return value


class ConfirmationRequest(BaseModel):
    """人工确认请求体。"""

    approved: bool = True
    tools: list[str] = Field(default_factory=list)


class ProfileUpdate(BaseModel):
    """用户画像更新请求。"""

    allergies: list[str] | None = None
    dislikes: list[str] | None = None
    nutrition_goals: list[str] | None = None
    inventory: list[str] | None = None


settings = get_settings()
tracing_enabled = configure_tracing(settings)
profile_store = UserProfileStore(settings.agent_db_path)
checkpointer, checkpoint_connection = create_sqlite_checkpointer(settings.checkpoint_db_path)
registry = create_registry()
planner = None
if settings.use_llm_planner and settings.llm_api_key:
    planner = LLMRecipePlanner(
        registry=registry,
        model_name=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
agent_graph = build_what_to_eat(
    registry,
    max_steps=settings.max_agent_steps,
    planner=planner,
    responder=create_responder(settings),
    checkpointer=checkpointer,
)
query_analyzer = create_query_analyzer(settings)

app = FastAPI(
    title="食谱规划 Agent",
    version="0.1.0",
    description="基于 LangGraph 的食谱规划 Agent。",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """将请求参数错误统一为客户端可解析的 JSON。"""

    logger.info("请求参数校验失败：%s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=422,
        content={
            "detail": "请求参数不合法",
            "error_type": "validation_error",
            "errors": jsonable_encoder(exc.errors()),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """隐藏内部堆栈，同时保证任何运行异常都有稳定 JSON 契约。"""

    logger.exception("未处理异常：%s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务执行失败，请稍后重试。",
            "error_type": type(exc).__name__,
        },
    )


app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """返回最小可用前端页面。"""

    static_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(static_path)


@app.get("/healthz")
def health_check() -> dict[str, str]:
    """返回服务健康状态和已注册工具名称。"""

    return {
        "status": "ok",
        "tools": ",".join(registry.names()),
        "neo4j": str(settings.use_neo4j).lower(),
        "hybrid": str(settings.use_hybrid).lower(),
        "file_search": str(settings.use_file_search).lower(),
        "llm_planner": str(settings.use_llm_planner and bool(settings.llm_api_key)).lower(),
        "llm_query_analyzer": str(
            settings.use_llm_query_analyzer and bool(settings.llm_api_key)
        ).lower(),
        "llm_responder": str(settings.use_llm_responder and bool(settings.llm_api_key)).lower(),
        "tracing": str(tracing_enabled).lower(),
        "trace_exporter": settings.otel_exporter if tracing_enabled else "disabled",
    }


def _plan_recipe(request: PlanRequest) -> dict:
    """运行 Agent 图，并返回客户端需要的最终状态字段。"""

    started = time.perf_counter()
    profile = profile_store.get(request.user_id)
    analyzer_started = time.perf_counter()
    with trace_span(
        "agent.query_analyzer",
        {"agent.goal_length": len(request.goal)},
    ) as span:
        query_plan = query_analyzer.analyze(request.goal, history=request.history)
        set_span_attributes(
            span,
            {
                "agent.intent": query_plan.get("intent"),
                "agent.entity_count": len(query_plan.get("entities") or []),
            },
        )
    analyzer_ms = round((time.perf_counter() - analyzer_started) * 1000, 2)
    constraints = build_effective_constraints(
        profile_constraints=profile_store.to_constraints(profile),
        request_constraints=dict(request.constraints),
        explicit_keys=set(request.explicit_constraints),
        query_plan=query_plan,
    )
    state = create_initial_state(
        goal=request.goal,
        constraints=constraints,
        user_id=request.user_id,
        conversation_id=request.conversation_id,
        history=request.history,
        meal_plan=request.meal_plan,
        shopping_list=request.shopping_list,
        confirmation_required=request.confirmation_required,
        query_plan=query_plan,
    )
    thread_id = request.thread_id or str(uuid4())
    result = agent_graph.invoke(
        state,
        config={"configurable": {"thread_id": thread_id}},
    )
    blocked_ids = set(result.get("blocked_recipe_ids", []))
    evidence = list(
        {
            item.recipe_id: item
            for item in result.get("evidence", [])
            if item.recipe_id not in blocked_ids
        }.values()
    )
    recipe_details = list(
        {item.recipe_id: item for item in result.get("recipe_details", [])}.values()
    )
    trace_events = [item.model_dump() for item in result.get("trace_events", [])]
    trace_events.insert(
        0,
        {
            "name": "query_analyzer",
            "duration_ms": analyzer_ms,
            "data": {"intent": query_plan.get("intent")},
        },
    )
    total_ms = round((time.perf_counter() - started) * 1000, 2)
    trace_events.append(
        {
            "name": "request",
            "duration_ms": total_ms,
            "data": {
                "status": result.get("status"),
                "tool_calls": len(result.get("tool_calls", [])),
            },
        }
    )
    return {
        "thread_id": thread_id,
        "conversation_id": request.conversation_id,
        "status": result.get("status"),
        "final_answer": result.get("final_answer"),
        "evidence": [item.model_dump() for item in evidence],
        "blocked_recipe_ids": list(blocked_ids),
        "recipe_details": [item.model_dump() for item in recipe_details],
        "pairing_recommendations": result.get("pairing_recommendations", []),
        "tool_calls": [item.model_dump() for item in result.get("tool_calls", [])],
        "validation_errors": result.get("validation_errors", []),
        "answer_comparison": result.get("answer_comparison"),
        "meal_plan": result.get("meal_plan"),
        "shopping_list": result.get("shopping_list", []),
        "pending_tool": (result.get("current_task") or {}).get("tool_name"),
        "query_plan": result.get("query_plan", {}),
        "search_strategy": result.get("search_strategy"),
        "query_analysis": result.get("query_analysis", {}),
        "trace": {
            "total_ms": total_ms,
            "events": trace_events,
        },
    }


@app.post("/plan")
def plan_recipe(request: PlanRequest) -> dict:
    """为每次请求建立顶层 span，并把 trace id 返回给调用方。"""

    fallback_trace_id = uuid4().hex
    with trace_span(
        "http.server.request",
        {
            "http.request.method": "POST",
            "url.path": "/plan",
            "agent.user_id": request.user_id,
            "agent.thread_id": request.thread_id,
        },
        kind="server",
    ) as span:
        response = _plan_recipe(request)
        response["trace"]["trace_id"] = current_trace_id(fallback_trace_id)
        set_span_attributes(
            span,
            {
                "agent.status": response.get("status"),
                "agent.thread_id": response.get("thread_id"),
                "agent.tool_calls": len(response.get("tool_calls") or []),
                "agent.evidence_count": len(response.get("evidence") or []),
                "agent.has_meal_plan": bool(response.get("meal_plan")),
                "agent.shopping_list_count": len(response.get("shopping_list") or []),
                "agent.total_ms": response.get("trace", {}).get("total_ms"),
            },
        )
        return response


@app.get("/tasks/{thread_id}")
def get_task(thread_id: str) -> dict:
    """读取指定任务的最新持久化状态。"""

    snapshot = agent_graph.get_state(config={"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        return {"thread_id": thread_id, "found": False}
    return {
        "thread_id": thread_id,
        "found": True,
        "values": snapshot.values,
        "next": snapshot.next,
    }


@app.post("/tasks/{thread_id}/confirm")
def confirm_task(thread_id: str, request: ConfirmationRequest) -> dict:
    """批准或拒绝暂停中的敏感工具调用，并恢复同一个任务。"""

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = agent_graph.get_state(config)
    if not snapshot.values:
        return {"thread_id": thread_id, "found": False}

    if not request.approved:
        return {
            "thread_id": thread_id,
            "found": True,
            "status": "cancelled",
            "final_answer": "用户拒绝了待执行的写操作，任务已取消。",
        }

    confirmed_tools = list(
        dict.fromkeys(
            [
                *snapshot.values.get("confirmed_tools", []),
                *request.tools,
            ]
        )
    )
    result = agent_graph.invoke(
        {
            "confirmation_required": True,
            "confirmed_tools": confirmed_tools,
            "planner_done": False,
            "status": "confirmed",
        },
        config=config,
    )
    blocked_ids = set(result.get("blocked_recipe_ids", []))
    evidence = list(
        {
            item.recipe_id: item
            for item in result.get("evidence", [])
            if item.recipe_id not in blocked_ids
        }.values()
    )
    return {
        "thread_id": thread_id,
        "found": True,
        "status": result.get("status"),
        "final_answer": result.get("final_answer"),
        "pending_tool": (result.get("current_task") or {}).get("tool_name"),
        "meal_plan": result.get("meal_plan"),
        "shopping_list": result.get("shopping_list", []),
        "tool_calls": [item.model_dump() for item in result.get("tool_calls", [])],
        "evidence": [item.model_dump() for item in evidence],
        "blocked_recipe_ids": list(blocked_ids),
    }


@app.get("/profiles/{user_id}", response_model=UserProfile)
def get_profile(user_id: str) -> UserProfile:
    """读取用户长期画像。"""

    return profile_store.get(user_id)


@app.put("/profiles/{user_id}", response_model=UserProfile)
def update_profile(user_id: str, request: ProfileUpdate) -> UserProfile:
    """更新用户长期画像。"""

    updates = {key: value for key, value in request.model_dump().items() if value is not None}
    return profile_store.merge(user_id, updates)
