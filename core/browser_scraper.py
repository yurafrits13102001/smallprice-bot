"""Headless-browser image scraping (Playwright).

Free fallback for marketplaces whose product image is injected by JavaScript and
is therefore absent from the static HTML curl_cffi sees — confirmed for 1688 (all
of it) and the bulk of AliExpress (a probe got fetch_ok 40/40 but imagePathList
0/40). A real browser renders the page, runs its JS, and we read the resulting
DOM. Runs on our own host, so there is no per-request cost; the only limits are
CPU/RAM, hence the intentionally low concurrency.

Optional proxy (PLAYWRIGHT_PROXY) for sites that block the server IP. AliExpress
renders fine direct; 1688 may need one.
"""
import asyncio
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
)

# Same extraction priority as the (now unusable) Apify pageFunction: meta image
# first, then the largest rendered <img> by pixel area (skips logos/icons).
_IMG_JS = """() => {
  const q = (s) => document.querySelector(s);
  let img = q('meta[property="og:image"]')?.content
    || q('meta[name="twitter:image"]')?.content
    || q('#landingImage')?.src
    || q('link[rel="image_src"]')?.href
    || '';
  if (!img) {
    let best = '', bestArea = 0;
    for (const im of document.querySelectorAll('img')) {
      const src = im.currentSrc || im.src || '';
      if (!src.startsWith('http')) continue;
      const area = (im.naturalWidth || 0) * (im.naturalHeight || 0);
      if (area > bestArea) { bestArea = area; best = src; }
    }
    img = best;
  }
  return img;
}"""


def _proxy_config(proxy: str) -> dict | None:
    """Parse a proxy string into Playwright's proxy dict.

    Accepts `host:port`, `http://host:port`, or `http://user:pass@host:port`
    (most residential proxies are authenticated). Playwright wants the server
    without credentials and username/password as separate fields.
    """
    if not proxy:
        return None
    p = urlparse(proxy if "://" in proxy else f"http://{proxy}")
    if not p.hostname:
        return None
    server = f"{p.scheme}://{p.hostname}" + (f":{p.port}" if p.port else "")
    cfg: dict = {"server": server}
    if p.username:
        cfg["username"] = p.username
    if p.password:
        cfg["password"] = p.password
    return cfg


def _is_1688(host: str) -> bool:
    return host == "1688.com" or host.endswith(".1688.com")


def _fetch_url(url: str) -> str:
    """1688's desktop detail page is browser-hostile; its mobile host renders lighter."""
    if _is_1688(urlparse(url).netloc.lower()):
        return url.replace("detail.1688.com", "m.1688.com")
    return url


async def scrape_images_playwright(
    urls: list[str],
    *,
    concurrency: int = 3,
    proxy: str = "",
    nav_timeout: float = 25.0,
    settle_ms: int = 3000,
) -> dict[str, str | None]:
    """Render each URL in headless Chromium → {original_url: image_url | None}.

    Keyed by the ORIGINAL url (1688 is fetched on its mobile host but mapped back).
    A single page that errors/times out maps to None instead of failing the batch.
    """
    if not urls:
        return {}
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        logger.error(f"Playwright unavailable ({e}); install with: playwright install --with-deps chromium")
        return {u: None for u in urls}

    out: dict[str, str | None] = {}
    sem = asyncio.Semaphore(max(1, concurrency))
    launch: dict = {
        "headless": True,
        # AutomationControlled flag is the easiest headless tell; dropping it (plus
        # the webdriver mask below) gets past the lighter anti-bot on AliExpress.
        "args": ["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
    }
    proxy_cfg = _proxy_config(proxy)
    if proxy_cfg:
        launch["proxy"] = proxy_cfg

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(**launch)

        async def one(url: str) -> None:
            host = urlparse(url).netloc.lower()
            ua = _MOBILE_UA if _is_1688(host) else _DESKTOP_UA
            async with sem:
                ctx = None
                try:
                    ctx = await browser.new_context(
                        user_agent=ua,
                        viewport={"width": 1280, "height": 1800},
                        locale="en-US",
                        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                    )
                    await ctx.add_init_script(
                        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
                    )
                    page = await ctx.new_page()
                    await page.goto(_fetch_url(url), wait_until="domcontentloaded",
                                    timeout=nav_timeout * 1000)
                    # Scroll partway to trigger lazy-loaded main images, then settle.
                    await page.wait_for_timeout(1200)
                    try:
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 3)")
                    except Exception:
                        pass
                    await page.wait_for_timeout(settle_ms)
                    img = await page.evaluate(_IMG_JS)
                    out[url] = img if isinstance(img, str) and img.startswith("http") else None
                except Exception as e:
                    logger.debug(f"playwright render failed {url[:70]}: {e}")
                    out[url] = None
                finally:
                    if ctx is not None:
                        try:
                            await ctx.close()
                        except Exception:
                            pass

        await asyncio.gather(*[one(u) for u in urls])
        await browser.close()

    found = sum(1 for v in out.values() if v)
    logger.info(f"Playwright: {found}/{len(urls)} images rendered")
    return out
