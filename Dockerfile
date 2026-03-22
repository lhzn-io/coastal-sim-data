FROM python:3.11-slim

WORKDIR /app

# Install system dev dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh

# Configure environment
ENV UV_PROJECT_ENVIRONMENT="/usr/local" \
    PYTHONPATH="/app/src"

COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY src/ /app/src/
COPY service/ /app/service/

RUN uv sync --frozen --no-dev --compile-bytecode

CMD ["uv", "run", "python", "-m", "coastal_data_serve.main", "--host", "0.0.0.0", "--port", "9598"]
