"""Firecrawl image scraping — a managed headless+proxy scraper for blocked sites.

The team's own handover doc recommends Firecrawl as the fix for the 403/captcha
wall on AliExpress and 1688 (which blocks the bot's direct scrape AND the admin
"AI Fill"). Firecrawl renders the page and rotates proxies server-side, getting
past blocks our curl_cffi / headless paths can't. A free-tier probe returned real
AliExpress og:images (2/3) and a real 1688 product page; the "stealth" proxy mode
does better on 1688 at a higher credit cost.

Returns {url: image_url | None}. Product photos on 1688/AliExpress live on the
Alibaba CDN under imgextra/ or kf/ paths; tfs/ paths are sprites/banners (incl.
the captcha banner), so we never accept those.
"""
import asyncio
import logging
import re

import httpx

logger = logging.getLogger(__name__)

_API = "https://api.firecrawl.dev/v1/scrape"

_PRODUCT_IMG_RE = re.compile(
    r'https?://(?:ae\d+|img)\.alicdn\.com/(?:imgextra|kf)/[^\s"\\\']+?\.(?:jpg|jpeg|png|webp)',
    re.IGNORECASE,
)
# Rendered-but-not-the-product pages (block/redirect/homepage) — reject by title
# so we don't index the captcha banner that sits on those pages.
_BAD_TITLE_RE = re.compile(r"captcha|阿里1688首页|访问验证|robot|^1688$", re.IGNORECASE)


def _pick_image(meta: dict, html: str) -> str | None:
    if _BAD_TITLE_RE.search(str(meta.get("title") or "")):
        return None
    og = meta.get("ogImage") or meta.get("og:image")
    if isinstance(og, str) and og.startswith("http") and "alicdn.com/tfs/" not in og:
        return og
    m = _PRODUCT_IMG_RE.search(html or "")
    return m.group(0) if m else None


async def _scrape_one(client: httpx.AsyncClient, api_key: str, url: str,
                      proxy: str, wait_ms: int) -> str | None:
    payload: dict = {"url": url, "formats": ["html"], "waitFor": wait_ms, "timeout": 45000}
    if proxy:
        payload["proxy"] = proxy
    try:
        r = await client.post(_API, json=payload, headers={"Authorization": f"Bearer {api_key}"})
    except Exception as e:
        logger.debug(f"firecrawl error {url[:60]}: {e}")
        return None
    if r.status_code != 200:
        logger.warning(f"firecrawl HTTP {r.status_code} {url[:60]}: {r.text[:120]}")
        return None
    try:
        data = r.json().get("data") or {}
    except Exception:
        return None
    return _pick_image(data.get("metadata") or {}, data.get("html") or "")


async def scrape_images_firecrawl(
    urls: list[str],
    api_key: str,
    *,
    concurrency: int = 4,
    proxy: str = "auto",
    wait_ms: int = 4000,
) -> dict[str, str | None]:
    """Resolve product images via Firecrawl. Each URL maps to image_url | None.

    proxy: "" (basic) | "auto" (escalate to stealth on block) | "stealth" (always,
    higher credit cost — best for 1688). A single failed URL maps to None.
    """
    if not urls or not api_key:
        return {}
    out: dict[str, str | None] = {}
    sem = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(timeout=70) as client:
        async def one(u: str) -> None:
            async with sem:
                out[u] = await _scrape_one(client, api_key, u, proxy, wait_ms)
        await asyncio.gather(*[one(u) for u in urls])
    found = sum(1 for v in out.values() if v)
    logger.info(f"Firecrawl: {found}/{len(urls)} images")
    return out
