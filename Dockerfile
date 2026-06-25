FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install apify-client

COPY bot/ bot/
COPY core/ core/
COPY scripts/ scripts/

RUN mkdir -p data

CMD ["python", "-m", "bot.main"]
