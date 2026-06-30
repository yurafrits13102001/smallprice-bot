"""Quick Firecrawl smoke test on a few AliExpress + 1688 URLs.

Confirms the key works and that Firecrawl gets real product images past the
captcha wall — and lets you compare FIRECRAWL_PROXY modes (auto vs stealth) on
1688 before spending credits on a full build. Pulls sample URLs from the index
meta.

Run:  python -m scripts.probe_firecrawl
"""
import asyncio
import logging
import pickle
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

from bot.config import settings
from core.firecrawl_scraper import scrape_images_firecrawl


def _sample(products, needle: str, n: int) -> list[str]:
    urls = [
        u for p in products for u in [p.link] + p.supplier_links
        if u and needle in urlparse(u).netloc.lower()
    ]
    step = max(len(urls) // n, 1)
    return urls[::step][:n]


async def main() -> None:
    if not settings.firecrawl_api_key:
        print("❗ FIRECRAWL_API_KEY не задано в .env")
        return
    products = pickle.load(open(f"{settings.index_path}.meta", "rb"))["products"]
    test = _sample(products, "1688", 5) + _sample(products, "aliexpress", 5)
    print(f"Testing {len(test)} URLs | proxy={settings.firecrawl_proxy!r} | concurrency={settings.firecrawl_concurrency}\n")
    res = await scrape_images_firecrawl(
        test, settings.firecrawl_api_key,
        concurrency=settings.firecrawl_concurrency, proxy=settings.firecrawl_proxy,
    )
    for u in test:
        img = res.get(u)
        print(f"{'IMG ' if img else 'NONE'} {urlparse(u).netloc:20} {(img or '')[:70]}")
    ok = sum(1 for v in res.values() if v)
    print(f"\n{ok}/{len(test)} resolved")


if __name__ == "__main__":
    asyncio.run(main())
