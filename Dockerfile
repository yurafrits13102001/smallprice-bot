FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && rm -rf /var/lib/apt/lists/*

# CPU-only torch (~190 MB) instead of the default CUDA build (~532 MB download,
# ~2.5 GB installed). The bot runs CLIP on CPU, so the GPU stack is dead weight.
# Installed first so the `torch` line in requirements.txt sees it satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install apify-client

COPY bot/ bot/
COPY core/ core/
COPY scripts/ scripts/

RUN mkdir -p data

CMD ["python", "-m", "bot.main"]
