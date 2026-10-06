# Docker 启动与排障

## 常见症状

- Docker API 返回 `500 Internal Server Error`
- `docker ps` 长时间无响应
- 日志出现 `connect tcp 192.168.65.7:2376: no route to host`
- 启动 Milvus 后 Docker Desktop 或 WSL 直接退出

## 根因

Windows 主机内存较小时，RAGFlow、Neo4j、Milvus 和 Docker Linux VM
会同时争抢内存，导致 WSL VM 或 Docker Engine 被系统终止。

## 配置

建议在 `C:\Users\<用户>\.wslconfig` 中限制并预留资源：

```ini
[wsl2]
memory=10GB
processors=4
swap=2GB

[experimental]
autoMemoryReclaim=gradual
```

不要让 RAGFlow 与 Neo4j、Milvus 在 16GB 内存机器上同时满载运行。

## 恢复步骤

```powershell
Get-Process | Where-Object { $_.ProcessName -match 'docker|com.docker' } |
  Stop-Process -Force -ErrorAction SilentlyContinue
wsl --shutdown
Start-Sleep -Seconds 8
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
```

等待 Docker 恢复后启动项目服务：

```powershell
docker compose up -d neo4j etcd minio milvus
.\scripts\run_dev.ps1
```

## 验证

```powershell
docker version
docker compose ps
curl.exe --noproxy "*" http://127.0.0.1:8020/healthz
```

健康检查中应包含：

```json
{"neo4j":"true","hybrid":"true","file_search":"false"}
```

Docker 不可用时仍可使用 [文件检索模式](../README.md)验证 Agent 和工具逻辑。
