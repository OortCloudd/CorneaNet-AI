# CorneaForge Dockerfile
#
# USAGE:
#   docker build -t corneaforge .
#   docker run -p 8000:8000 corneaforge

FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
COPY src/ src/

RUN uv sync --frozen --no-dev --extra server

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "corneaforge.server:app", "--host", "0.0.0.0", "--port", "8000"]
