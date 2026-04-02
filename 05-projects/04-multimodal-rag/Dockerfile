FROM astral/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY . .

EXPOSE 8000 8501

RUN chmod +x ./scripts/start.sh

CMD ["./scripts/start.sh"]
