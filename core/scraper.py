import re
import logging
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
    return any(host == d or host.endswith("." + d) for d in SUPPORTED_DOMAINS)


def _clean_title(title: str) -> str:
    for pattern, replacement in TITLE_CLEANERS:
        title = re.sub(pattern, replacement, title, flags=re.IGNORECASE)
    title = title.strip()
    if re.match(r"^(amazon|aliexpress|temu|tiktok|1688)\.[a-z.]*$", title, re.IGNORECASE):
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
        for key in ("title", "name", "product"):
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


async def scrape_product_title(url: str) -> str | None:
    host = _get_domain(url)
    headers = HEADERS_1688 if "1688.com" in host else HEADERS

    soup = None
    try:
        async with httpx.AsyncClient(
            headers=headers,
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
        # Level 1: og:title
        og = soup.find("meta", property="og:title")
        if og and og.get("content", "").strip():
            cleaned = _clean_title(og["content"].strip())
            if cleaned:
                return cleaned

        # Level 2: <title>
        if soup.title and soup.title.string:
            cleaned = _clean_title(soup.title.string.strip())
            if cleaned:
                return cleaned

        # Level 3: CSS selectors (site-specific first, then generic)
        for selector in _CSS_SELECTORS:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(strip=True)
                if len(text) > 3:
                    return text

    # Level 4: extract from URL path (eBay and others embed title in URL)
    url_title = _title_from_url(url)
    if url_title:
        return url_title

    return None