import re
import logging
import asyncio
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

HEADERS_1688 = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

TITLE_CLEANERS = [
    (r"\s*[-|–—]\s*AliExpress.*$", ""),
    (r"^AliExpress\s*[-|–—]\s*", ""),
    (r"\s*[-|–—]\s*Temu.*$", ""),
    (r"^Temu\s*[-|–—]\s*", ""),
    (r"^Amazon\.[a-z.]+\s*[:\-–—]\s*", ""),
    (r"\s*[-|–—]\s*Amazon\.[a-z.]+.*$", ""),
    (r"\s*[:\-–—]\s*[Кк]упить.*$", ""),
    (r"\s*\|\s*TikTok.*$", ""),
    (r"^1688\.com\s*[-|–—]\s*", ""),
    (r"\s*[-–—]\s*1688\.com.*$", ""),
    (r"\s*[-–—]\s*Интернет-магазин.*$", ""),
    (r"\s*[-–—]\s*阿里巴巴.*$", ""),
    (r"^【.*?】\s*", ""),
]

HARD_DOMAINS = ["ebay.com"]

SKIP_APIFY_DOMAINS = ["1688.com"]

SUPPORTED_DOMAINS = [
    "aliexpress.com",
    "amazon.com", "amazon.co.uk", "amazon.de", "amazon.it",
    "amazon.in", "amazon.ca", "amazon.co.jp", "amazon.com.au",
    "amazon.ae", "amazon.sa",
    "temu.com",
    "tiktok.com",
    "1688.com",
]


_BLOCKED_RE = re.compile(
    r"access.?denied|captcha|robot.?check|are you a human|verify you.?re human|"
    r"ddos.?guard|security check|please enable cookies",
    re.IGNORECASE,
)

_CSS_SELECTORS = [
    "#productTitle",                    # Amazon
    "h1.d-title",                       # 1688
    ".mod-detail-title",                # 1688
    "h1",                               # generic
    ".product-title",                   # generic
    ".product-name h1",                 # generic
    "[data-testid='product-title']",    # various
]


def _get_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def is_supported_url(url: str) -> bool:
    host = _get_domain(url)
    if not host:
        return False
    if re.match(r'^amazon\.[a-z.]+$', host):
        return True
    return any(host == d or host.endswith("." + d) for d in SUPPORTED_DOMAINS)


def _clean_title(title: str) -> str:
    for pattern, replacement in TITLE_CLEANERS:
        title = re.sub(pattern, replacement, title, flags=re.IGNORECASE)
    title = title.strip()
    if re.match(r"^(amazon|aliexpress|temu|tiktok|1688)\.[a-z.]*$", title, re.IGNORECASE):
        return ""
    if _BLOCKED_RE.search(title):
        return ""
    return title


def _is_blocked(soup: BeautifulSoup) -> bool:
    title_tag = soup.title
    title_text = (title_tag.string or "") if title_tag else ""
    body_snippet = soup.get_text()[:500]
    return bool(_BLOCKED_RE.search(f"{title_text} {body_snippet}"))


def _title_from_url(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        for key in ("title", "name", "product", "_skw"):
            vals = qs.get(key)
            if vals:
                candidate = re.sub(r"[-_]+", " ", vals[0]).strip()
                if len(candidate) > 5 and re.search(r"[a-zA-Z]{3,}", candidate):
                    return candidate

        parts = [p for p in parsed.path.split("/") if p]
        for part in reversed(parts):
            if part.isdigit() or len(part) < 10:
                continue
            if "-" in part or "_" in part:
                candidate = re.sub(r"[-_]+", " ", part).strip()
                if re.search(r"[a-zA-Z]{3,}", candidate):
                    return candidate
    except Exception:
        pass
    return None


def _scrape_1688_mobile(url: str) -> str | None:
    try:
        from curl_cffi import requests as cffi_requests
        mobile_url = url.replace("detail.1688.com", "m.1688.com")
        r = cffi_requests.get(mobile_url, impersonate="chrome120", timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        if _is_blocked(soup):
            return None
        if soup.title and soup.title.string:
            cleaned = _clean_title(soup.title.string.strip())
            if cleaned:
                return cleaned
    except Exception as e:
        logger.warning(f"1688 mobile scrape failed for {url}: {e}")
    return None


def _scrape_with_apify(url: str, apify_token: str) -> str | None:
    try:
        from apify_client import ApifyClient
        client = ApifyClient(apify_token)
        run_input = {
            "startUrls": [{"url": url}],
            "pageFunction": (
                "async function pageFunction(context) {\n"
                "    await new Promise(r => setTimeout(r, 3000));\n"
                "    const ogTitle = document.querySelector('meta[property=\"og:title\"]')?.content || '';\n"
                "    const title = document.title || '';\n"
                "    const h1 = document.querySelector('h1')?.innerText || '';\n"
                "    return { ogTitle, title, h1 };\n"
                "}"
            ),
            "proxyConfiguration": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]},
        }
        run = client.actor("apify/web-scraper").call(run_input=run_input)
        dataset_id = run.default_dataset_id
        logger.info(f"Apify run finished, dataset_id={dataset_id}")
        for item in client.dataset(dataset_id).iterate_items():
            logger.info(f"Apify item: {item}")
            for key in ("ogTitle", "title", "h1"):
                raw = item.get(key, "").strip()
                if raw:
                    cleaned = _clean_title(raw)
                    if cleaned:
                        return cleaned
    except Exception as e:
        logger.warning(f"Apify scrape failed for {url}: {e}")
    return None


async def normalize_title(client: AsyncOpenAI, title: str) -> str:
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract short product names from marketplace titles. "
                        "Return ONLY the short product name in English, 3-7 words. "
                        "Remove sizes, colors, quantities, marketing words, seller names. "
                        "Keep brand name if present. Keep key product identifiers.\n"
                        "Examples:\n"
                        "Input: 'BUDI Multifunctional 9 in 1 Data Cable with USB Type-C Card Reader with Sim Card Tray Pin Eject Removal Tool Needle Opener Ejector with SIM/TF Card Storage Small Box : Cell Phones & Accessories'\n"
                        "Output: 'BUDI 9-in-1 multifunctional adapter'\n"
                        "Input: 'Sixfire Gas Stove Windproof Energy Saving Camping Stainless Steel Gas Stove Ultra Light Folding Furnace Outdoor Metal Camping Equipment'\n"
                        "Output: 'Ultralight camping gas stove'\n"
                        "Input: '2024 New Arrivals Car Phone Holder Mount Dashboard Windshield Air Vent Universal Cell Phone Holder for Car'\n"
                        "Output: 'Universal car phone holder mount'"
                    ),
                },
                {"role": "user", "content": title},
            ],
        )
        result = response.choices[0].message.content.strip().strip("'\"")
        logger.info(f"Normalized: '{title[:60]}...' -> '{result}'")
        return result
    except Exception as e:
        logger.warning(f"Normalization failed: {e}")
        return title


async def scrape_product_image_url(url: str) -> str | None:
    """Extract main product image URL (og:image or Amazon landingImage)."""
    try:
        async with httpx.AsyncClient(
            headers=HEADERS, follow_redirects=True, timeout=10.0
        ) as client:
            resp = await client.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        if _is_blocked(soup):
            return None
        og = soup.find("meta", property="og:image")
        if og and str(og.get("content", "")).startswith("http"):
            return str(og["content"])
        img = soup.find("img", {"id": "landingImage"})
        if img:
            for attr in ("data-old-hires", "src"):
                val = str(img.get(attr, ""))
                if val.startswith("http"):
                    return val
    except Exception as e:
        logger.warning(f"Image scrape failed for {url}: {e}")
    return None


async def scrape_product_title_fast(url: str) -> str | None:
    host = _get_domain(url)
    is_hard = any(host == d or host.endswith("." + d) for d in HARD_DOMAINS)
    is_1688 = host == "1688.com" or host.endswith(".1688.com")

    if is_1688:
        result = await asyncio.to_thread(_scrape_1688_mobile, url)
        if result:
            return result

    if not is_hard:
        soup = None
        try:
            async with httpx.AsyncClient(
                headers=HEADERS,
                follow_redirects=True,
                timeout=15.0,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except httpx.TimeoutException:
            logger.warning(f"Timeout: {url}")
        except httpx.HTTPError as e:
            logger.warning(f"HTTP error for {url}: {e}")

        if soup and not _is_blocked(soup):
            og = soup.find("meta", property="og:title")
            if og and og.get("content", "").strip():
                cleaned = _clean_title(og["content"].strip())
                if cleaned:
                    return cleaned

            if soup.title and soup.title.string:
                cleaned = _clean_title(soup.title.string.strip())
                if cleaned:
                    return cleaned

            for selector in _CSS_SELECTORS:
                el = soup.select_one(selector)
                if el:
                    text = el.get_text(strip=True)
                    if len(text) > 3:
                        return text

    return _title_from_url(url)


async def scrape_product_title_apify(url: str, apify_token: str) -> str | None:
    host = _get_domain(url)
    if any(host == d or host.endswith("." + d) for d in SKIP_APIFY_DOMAINS):
        return None
    return await asyncio.to_thread(_scrape_with_apify, url, apify_token)