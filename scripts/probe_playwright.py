"""Quick Playwright smoke test on a handful of AliExpress + 1688 URLs.

Run this BEFORE a full rebuild to confirm the headless browser renders product
images — and, crucially, whether 1688 needs a proxy (set PLAYWRIGHT_PROXY). Pulls
representative sample URLs from the index meta.

Run:  python -m scripts.probe_playwright
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
from core.browser_scraper import scrape_images_playwright


def _sample(products, needle: str, n: int) -> list[str]:
    urls = [
        u for p in products for u in [p.link] + p.supplier_links
        if u and needle in urlparse(u).netloc.lower()
    ]
    step = max(len(urls) // n, 1)
    return urls[::step][:n]


async def main() -> None:
    products = pickle.load(open(f"{settings.index_path}.meta", "rb"))["products"]
    test = _sample(products, "aliexpress", 5) + _sample(products, "1688", 5)
    print(
        f"Testing {len(test)} URLs | concurrency={settings.playwright_concurrency} "
        f"| proxy={'set' if settings.playwright_proxy else 'none'}\n"
    )
    res = await scrape_images_playwright(
        test,
        concurrency=settings.playwright_concurrency,
        proxy=settings.playwright_proxy,
    )
    for u in test:
        img = res.get(u)
        print(f"{'IMG ' if img else 'NONE'} {urlparse(u).netloc:22} {(img or '')[:70]}")
    ok = sum(1 for v in res.values() if v)
    print(f"\n{ok}/{len(test)} rendered")


if __name__ == "__main__":
    asyncio.run(main())
