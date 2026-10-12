# 云服务器部署

这套服务由 5 个容器组成（backend + Neo4j + Milvus + etcd + MinIO），需要一台常开、
内存够用的 Linux 服务器。下面的规格和参数都是按本机实测占用写的，不是拍脑袋估的。

前置条件：Docker 20.10 以上，且 **Docker Compose 2.24 以上**（覆盖文件用到了 `!override`
语法，脚本会自己检查并提示）。

## 一、买什么机器

本机 `docker stats` 实测常驻占用：

| 容器 | 常驻内存 |
|---|---:|
| backend（含 BGE 向量模型） | 676 MB |
| Neo4j | 506 MB |
| Milvus | 156 MB |
| MinIO | 91 MB |
| etcd | 28 MB |
| **合计** | **约 1.4 GB** |

单次混合检索还要加上模型推理的临时开销，Milvus 建索引时也会有峰值。

- 最低配置：**2 核 4 GB**，系统盘 **≥ 40 GB**（基础镜像加起来约 6.5 GB）
- 推荐配置：**2 核 8 GB**，Milvus 建索引时更稳
- 4 GB 的机器建议开 2 GB swap，避免 OOM 把 Milvus 或 Neo4j 杀掉
- 安全组只放行 `22` 和 Web 端口（默认 `8020`）。**不要**放行 `7687`、`19530`、`9000`，
  这几个端口在 `deploy/compose.server.yml` 里已经只绑到 `127.0.0.1`，公网访问不到

用"IP:端口"访问不需要备案；要绑域名就得先做 ICP 备案。

## 二、国内网络怎么把镜像拉下来

Docker Hub 和 PyPI 在国内常常很慢，两条线路任选一条。

线路一：在服务器上配镜像加速，然后直接构建

```bash
# Docker 镜像加速（示例，换成你云厂商提供的地址，腾讯云/阿里云控制台都能拿到）
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<'EOF'
{ "registry-mirrors": ["https://<你的镜像加速地址>"] }
EOF
sudo systemctl restart docker

# pip 国内源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

`Dockerfile` 里 torch 已经指定走 CPU 版索引；如果 `download.pytorch.org` 太慢，
可以把它换成上海交大的镜像 `https://mirror.sjtu.edu.cn/pytorch-wheels/cpu`。

线路二（最稳）：在本地把镜像打好再搬过去

```powershell
docker save what-to-eat-backend:latest neo4j:5.18 milvusdb/milvus:v2.5.11 `
  quay.io/coreos/etcd:v3.5.18 minio/minio:RELEASE.2023-03-20T20-16-18Z `
  -o images.tar   # 约 6 GB
scp images.tar root@<服务器IP>:/root/
```

```bash
docker load -i /root/images.tar
BUILD=0 bash deploy/deploy.sh
```

## 三、部署步骤

### 1. 服务器上拉代码

```bash
git clone https://github.com/xie-zp/what-to-eat.git /opt/what-to-eat
cd /opt/what-to-eat
mkdir -p data/generated
```

### 2. 上传菜谱数据（本地生成的两份 json）

原始菜谱数据不随仓库分发，但生成结果可以直接带过去，保证线上和你本地评测用的是同一份数据。

```powershell
scp data\generated\recipes.json data\generated\graph.json `
  root@<服务器IP>:/opt/what-to-eat/data/generated/
```

### 3. 上传 BGE 向量模型（92 MB）

服务器上如果没有模型缓存，容器会因为 `HF_HUB_OFFLINE=1` 直接失败。先把本机已下好的模型传过去：

```powershell
scp -r "$env:USERPROFILE\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5" `
  root@<服务器IP>:/opt/what-to-eat/.runtime/huggingface/hub/
```

不想传也行：把 `deploy/compose.server.yml` 里注释掉的三行 `HF_HUB_OFFLINE` /
`TRANSFORMERS_OFFLINE` / `HF_ENDPOINT` 打开，让容器自己从 `hf-mirror.com` 下载。

### 4. 配置 .env

```bash
cp .env.example .env
```

至少要改两个值：

```
LLM_API_KEY=你的密钥
NEO4J_PASSWORD=自己换一个强密码
HF_CACHE_PATH=/opt/what-to-eat/.runtime/huggingface
```

`NEO4J_PASSWORD` 只在**第一次启动前**设置有效：如果容器已经用旧密码初始化过数据库，
改密码需要先 `docker compose down -v` 清掉卷再重新导入。

### 5. 一键部署

```bash
bash deploy/deploy.sh
```

脚本会依次做：构建镜像并起容器 → 等 Neo4j/Milvus 就绪 → 导入知识图谱 →
构建向量索引 → 重启后端 → 健康检查。跑完打开 `http://<服务器公网IP>:8020`。

## 四、上线后要注意

- 网址一公开，谁都能点。`.env` 里的 DeepSeek key 会被陌生人消耗，建议单独申请一个
  限额 key；不想花这个钱就把 `USE_LLM_PLANNER` / `USE_LLM_QUERY_ANALYZER` /
  `USE_LLM_RESPONDER` 改成 `false`，Agent 会走确定性 Planner，功能链路照常。
- 镜像构建会留下构建缓存，磁盘紧张时执行 `docker system prune -af`。
- 更新代码后重新执行 `bash deploy/deploy.sh` 即可，数据文件不会被覆盖。
