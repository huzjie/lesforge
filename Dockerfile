FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY lesforge ./lesforge
COPY configs ./configs

RUN pip install --no-cache-dir .

EXPOSE 8000

ENV LESFORGE_CONFIG=/app/configs/mobile.yaml

CMD ["python", "-m", "lesforge.cli", "serve", "--config", "/app/configs/mobile.yaml"]
