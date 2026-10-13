FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 国内服务器构建时可以把镜像源换成内网源，例如
# PIP_INDEX_URL=http://mirrors.tencentyun.com/pypi/simple
# 见 deploy/README.md。
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

WORKDIR /app

COPY pyproject.toml ./

RUN pip install --no-cache-dir --timeout 180 --retries 8 torch \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --timeout 180 --retries 8 ".[rag]"

COPY backend ./backend
COPY scripts ./scripts
COPY README.md LICENSE ./

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
