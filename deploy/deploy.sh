#!/usr/bin/env bash
# 在云服务器上部署本项目：构建镜像 -> 起容器 -> 导入图谱 -> 建向量索引 -> 健康检查。
#
# 用法（在仓库根目录执行）：
#   bash deploy/deploy.sh
#   BUILD=0 bash deploy/deploy.sh   # 镜像已经在服务器上（docker load 过），跳过构建
#
# 重复执行是安全的：只重建容器和索引，不会碰 data/generated 下的数据文件。
set -euo pipefail

APP_DIR="${APP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WEB_PORT="${WEB_PORT:-8020}"
BUILD="${BUILD:-1}"
COMPOSE_FILES=(-f docker-compose.yml -f deploy/compose.server.yml)

info() { printf '\033[1;32m[deploy]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[deploy]\033[0m %s\n' "$*" >&2; exit 1; }

cd "$APP_DIR"

command -v docker >/dev/null 2>&1 || fail "没有检测到 Docker，先执行：curl -fsSL https://get.docker.com | sh"
docker compose version >/dev/null 2>&1 || fail "缺少 docker compose 插件，请升级 Docker 到 20.10 以上"

compose_version="$(docker compose version --short 2>/dev/null | sed 's/^v//')"
compose_major="${compose_version%%.*}"
compose_minor="$(printf '%s' "$compose_version" | cut -d. -f2)"
if [ "${compose_major:-0}" -lt 2 ] || { [ "${compose_major:-0}" -eq 2 ] && [ "${compose_minor:-0}" -lt 24 ]; }; then
  fail "docker compose 版本过低（${compose_version:-未知}），覆盖文件用到 !override，需要 2.24 以上，升级 Docker 即可"
fi

mkdir -p data/generated
[ -f data/generated/recipes.json ] || fail "缺少 data/generated/recipes.json：在本地跑完 scripts/parse_howtocook.py 后上传"
[ -f data/generated/graph.json ] || fail "缺少 data/generated/graph.json：在本地跑完 scripts/build_graph.py 后上传"

if [ ! -f .env ]; then
  cp .env.example .env
  info "已从 .env.example 生成 .env，请填好 LLM_API_KEY 和 NEO4J_PASSWORD 后重新执行本脚本"
  exit 0
fi

if ! grep -qE '^LLM_API_KEY=.+' .env; then
  info "提醒：.env 里还没有 LLM_API_KEY，Agent 会用规则 Planner 兜底（功能可用，只是没有大模型改写）"
fi

if ! grep -qE '^HF_CACHE_PATH=' .env; then
  info "提醒：.env 里没有 HF_CACHE_PATH，容器将使用仓库内 .runtime/huggingface 作为模型缓存"
fi

if [ "$BUILD" = "1" ]; then
  info "构建镜像并启动容器（首次构建要下载 torch 和基础镜像，5-15 分钟）"
  WEB_PORT="$WEB_PORT" docker compose "${COMPOSE_FILES[@]}" up -d --build
else
  info "跳过构建，直接用服务器上已有的镜像启动容器"
  WEB_PORT="$WEB_PORT" docker compose "${COMPOSE_FILES[@]}" up -d
fi

wait_healthy() {
  local container="$1" seconds="${2:-240}" waited=0 status
  while [ "$waited" -lt "$seconds" ]; do
    status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}started{{end}}' "$container" 2>/dev/null || echo missing)"
    [ "$status" = "healthy" ] && return 0
    sleep 5
    waited=$((waited + 5))
  done
  return 1
}

info "等待 Neo4j 就绪"
wait_healthy what-to-eat-neo4j 300 || fail "Neo4j 没有在 5 分钟内变成 healthy，执行 docker compose logs neo4j 查看原因"

info "等待 Milvus 就绪"
milvus_ready=0
for _ in $(seq 1 36); do
  if docker compose "${COMPOSE_FILES[@]}" exec -T backend \
    python -c "import urllib.request;urllib.request.urlopen('http://milvus:9091/healthz',timeout=5)" >/dev/null 2>&1; then
    milvus_ready=1
    break
  fi
  sleep 5
done
[ "$milvus_ready" = "1" ] || fail "Milvus 没有在 3 分钟内就绪，执行 docker compose logs milvus 查看原因"

info "导入知识图谱到 Neo4j"
docker compose "${COMPOSE_FILES[@]}" exec -T -e PYTHONPATH=/app/backend backend \
  python scripts/import_neo4j.py --reset

info "构建 Milvus 向量索引"
docker compose "${COMPOSE_FILES[@]}" exec -T -e PYTHONPATH=/app/backend backend \
  python scripts/build_vector_index.py

info "重启后端，让它接管刚建好的索引"
docker compose "${COMPOSE_FILES[@]}" restart backend >/dev/null

info "健康检查"
if ! command -v curl >/dev/null 2>&1; then
  info "服务器上没有 curl，跳过自动健康检查，直接访问 http://<服务器公网IP>:${WEB_PORT} 确认"
  exit 0
fi
for _ in $(seq 1 24); do
  if curl -fsS "http://127.0.0.1:${WEB_PORT}/healthz" >/dev/null 2>&1; then
    info "部署完成，打开 http://<服务器公网IP>:${WEB_PORT} 即可访问"
    exit 0
  fi
  sleep 5
done
fail "健康检查失败，执行 docker compose logs backend 查看原因"
