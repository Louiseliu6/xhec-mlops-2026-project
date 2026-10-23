FROM python:3.11-slim

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv==0.12.20

# Install locked runtime dependencies first so code changes reuse this cached layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --python 3.11

COPY src ./src
COPY data ./data
COPY bin ./bin

# Bake a freshly trained pipeline into the image; training is seeded, so it is reproducible.
RUN python -m src.modelling.main data/abalone.csv

# The UI is opened from the host on port 4200 (docker run -p 4200:4201), so its API URL
# must use the host port.
ENV PREFECT_HOME=/app/.prefect \
    PREFECT_SERVER_ANALYTICS_ENABLED=false \
    PREFECT_API_URL=http://127.0.0.1:4201/api \
    PREFECT_UI_API_URL=http://localhost:4200/api

EXPOSE 8001 4201
CMD ["bash", "bin/run_services.sh"]
