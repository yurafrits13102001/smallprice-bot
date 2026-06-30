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

# Sentinel: a TRANSIENT failure (timeout / 5xx / rate-limit / out-of-credits).
# The caller must NOT cache these — they should be retried on the next build,
# unlike a definitive None (page fetched, genuinely no product image / captcha).
_RETRY = object()
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}

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
                      proxy: str, wait_ms: int):
    """Returns image_url (str) | None (definitive: no product image) | _RETRY
    (transient failure — don't cache, retry next build)."""
    # Skip non-URL junk (e.g. "Main Link" text in a catalog cell) — it would just
    # waste an API call on a guaranteed 400. Definitive: cache as None.
    if not isinstance(url, str) or not url.startswith("http"):
        return None
    payload: dict = {"url": url, "formats": ["html"], "waitFor": wait_ms, "timeout": 60000}
    if proxy:
        payload["proxy"] = proxy
    try:
        r = await client.post(_API, json=payload, headers={"Authorization": f"Bearer {api_key}"})
    except Exception as e:
        logger.debug(f"firecrawl network error {url[:60]}: {e}")
        return _RETRY
    if r.status_code in _TRANSIENT_STATUS or r.status_code == 402:
        # 408 timeout / 5xx / 429 rate-limit / 402 out-of-credits — all retriable.
        logger.warning(f"firecrawl transient HTTP {r.status_code} {url[:60]}")
        return _RETRY
    if r.status_code != 200:
        # 400 junk URL, 401 bad key, 403 hard block — definitive, cache as None.
        logger.warning(f"firecrawl HTTP {r.status_code} {url[:60]}: {r.text[:100]}")
        return None
    try:
        data = r.json().get("data") or {}
    except Exception:
        return None
    return _pick_image(data.get("metadata") or {}, data.get("html") or "")


# Public alias so callers can test a single result against the transient sentinel.
RETRY = _RETRY


async def scrape_one_firecrawl(client: httpx.AsyncClient, api_key: str, url: str,
                               *, proxy: str = "auto", wait_ms: int = 4000):
    """Scrape ONE URL (reusing a shared client) for first-hit-per-product use.
    Returns image_url (str) | None (definitive miss) | RETRY (transient failure)."""
    return await _scrape_one(client, api_key, url, proxy, wait_ms)


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
    async with httpx.AsyncClient(timeout=90) as client:
        async def one(u: str) -> None:
            async with sem:
                res = await _scrape_one(client, api_key, u, proxy, wait_ms)
                # Omit transient failures so the caller leaves them uncached and
                # retries them next build (only definitive results are returned).
                if res is not _RETRY:
                    out[u] = res
        await asyncio.gather(*[one(u) for u in urls])
    found = sum(1 for v in out.values() if v)
    transient = len(urls) - len(out)
    logger.info(f"Firecrawl: {found}/{len(urls)} images ({transient} transient, will retry)")
    return out
