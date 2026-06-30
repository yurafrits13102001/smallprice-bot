"""Offline CLIP image-index build.

`scripts.build_index` builds only the TEXT index. The CLIP image index was
previously built solely inside the bot on .xlsx upload — a background task that a
container restart silently kills, which makes a full rebuild hard to run and
observe. This runs the SAME `build_clip_index_async` in the FOREGROUND: it
survives as long as the process does, prints progress, and reuses
`data/clip_image_cache.json`.

Products are read from the existing index meta (no .xlsx needed). To retry URLs
that failed before, purge the null entries from the cache first:

    python - <<'PY'
    import json; c=json.load(open("data/clip_image_cache.json"))
    json.dump({k:v for k,v in c.items() if v}, open("data/clip_image_cache.json","w"))
    PY

Run (inside the container, ideally under tmux/screen so an SSH drop can't kill it):

    python -m scripts.build_clip_index
"""
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# httpx logs every HTTP request at INFO — far too noisy for a multi-thousand-URL run.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("build_clip_index")

from bot.config import settings
from core.matcher import ProductMatcher


async def main() -> None:
    matcher = ProductMatcher(api_key=settings.openai_api_key)
    matcher.load_index(settings.index_path)  # loads products + url maps + any existing CLIP index
    products = matcher.products
    logger.info(
        f"CLIP build for {len(products)} products "
        f"(concurrency={settings.clip_scrape_concurrency}, timeout={settings.clip_scrape_timeout}s, "
        f"apify={'on' if settings.apify_token else 'off'})"
    )
    await matcher.build_clip_index_async(
        products,
        settings.index_path,
        settings.apify_token,
        concurrency=settings.clip_scrape_concurrency,
        scrape_timeout=settings.clip_scrape_timeout,
        firecrawl_api_key=settings.firecrawl_api_key,
        firecrawl_proxy=settings.firecrawl_proxy,
        firecrawl_concurrency=settings.firecrawl_concurrency,
        use_playwright=settings.use_playwright,
        playwright_concurrency=settings.playwright_concurrency,
        playwright_proxy=settings.playwright_proxy,
    )
    n = matcher.clip_index.index.ntotal if matcher.clip_index and matcher.clip_index.index else 0
    logger.info(f"Done. CLIP index now has {n} images. Run scripts.diagnose_images to see coverage.")


if __name__ == "__main__":
    asyncio.run(main())
