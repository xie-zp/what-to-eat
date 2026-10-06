FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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
