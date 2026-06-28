import re
import json
import logging
import asyncio
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


def _fetch_html_sync(url: str, timeout: float, headers: dict) -> tuple[str | None, str | None]:
    """Fetch page HTML with a real-browser TLS fingerprint (curl_cffi).

    Returns (html, error_key). Marketplaces block plain httpx by TLS fingerprint,
    so we impersonate Chrome for every domain — this is what defeats most bot walls.
    """
    from curl_cffi import requests as cffi_requests
    try:
        r = cffi_requests.get(
            url, impersonate="chrome120", headers=headers,
            timeout=timeout, allow_redirects=True,
        )
    except Exception as e:
        msg = str(e).lower()
        err = "timeout" if ("timeout" in msg or "timed out" in msg) else "network"
        logger.warning(f"fetch failed for {url}: {e}")
        return None, err
    if r.status_code >= 400:
        logger.warning(f"fetch HTTP {r.status_code} for {url}")
        return None, "network"
    return r.text, None


async def _fetch_soup(url: str, timeout: float, headers: dict) -> tuple[BeautifulSoup | None, str | None]:
    html, err = await asyncio.to_thread(_fetch_html_sync, url, timeout, headers)
    if html is None:
        return None, err
    return BeautifulSoup(html, "html.parser"), None


def _first_image(image) -> str | None:
    """Pull the first http(s) image URL out of a schema.org `image` field (str/dict/list)."""
    if isinstance(image, str):
        return image if image.startswith("http") else None
    if isinstance(image, dict):
        url = image.get("url")
        return url if isinstance(url, str) and url.startswith("http") else None
    if isinstance(image, list):
        for item in image:
            url = _first_image(item)
            if url:
                return url
    return None


def _extract_jsonld(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    """Return (name, image_url) from a schema.org/Product JSON-LD block, if present.

    This is marketplace-agnostic and more reliable than og:title/<title>, which
    often carry marketing junk or a block-page message.
    """
    import json as _json
    for tag in soup.find_all("script", type="application/ld+json"):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = _json.loads(raw)
        except Exception:
            continue
        if isinstance(data, list):
            nodes = data
        elif isinstance(data, dict):
            nodes = data.get("@graph", [data])
        else:
            continue
        for node in nodes:
            if not isinstance(node, dict):
                continue
            types = node.get("@type")
            types = types if isinstance(types, list) else [types]
            if "Product" not in types:
                continue
            name = node.get("name")
            img = _first_image(node.get("image"))
            name = name.strip() if isinstance(name, str) and name.strip() else None
            if name or img:
                return name, img
    return None, None


def _parse_title(soup: BeautifulSoup) -> str | None:
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
    return None


def _image_from_amazon(soup: BeautifulSoup) -> str | None:
    """Amazon main image. Prefers data-a-dynamic-image (a {url: [w,h]} map) → widest."""
    img = soup.find("img", {"id": "landingImage"})
    if not img:
        return None
    dyn = img.get("data-a-dynamic-image")
    if dyn:
        try:
            variants = json.loads(dyn)  # {"https://...jpg": [w, h], ...}
            best = max(
                variants.items(),
                key=lambda kv: kv[1][0] if isinstance(kv[1], list) and kv[1] else 0,
                default=(None, None),
            )[0]
            if best and str(best).startswith("http"):
                return str(best)
        except Exception:
            pass
    for attr in ("data-old-hires", "src"):
        val = str(img.get(attr, "")).strip()
        if val.startswith("http"):
            return val
    return None


# AliExpress/Temu/etc. often hide the main image in inline JSON, not in og:image.
_SCRIPT_IMG_RE = re.compile(
    r'"(?:imagePathList|summImagePathList)"\s*:\s*\[\s*"(https?:[^"]+)"'
    r'|"(?:imageUrl|mainImage|hdThumbUrl|imgUrl)"\s*:\s*"(https?:[^"]+?\.(?:jpg|jpeg|png|webp)[^"]*)"',
    re.IGNORECASE,
)


def _image_from_scripts(soup: BeautifulSoup) -> str | None:
    for tag in soup.find_all("script"):
        text = tag.string or tag.get_text() or ""
        if not text or "http" not in text:
            continue
        m = _SCRIPT_IMG_RE.search(text)
        if m:
            url = (m.group(1) or m.group(2) or "").replace("\\u002F", "/").replace("\\/", "/")
            if url.startswith("http"):
                return url
    return None


def _parse_image(soup: BeautifulSoup) -> str | None:
    # 1. Standard social/meta image tags — present on most marketplaces & shops.
    for attr, key in (
        ("property", "og:image"),
        ("property", "og:image:secure_url"),
        ("property", "og:image:url"),
        ("name", "twitter:image"),
        ("name", "twitter:image:src"),
    ):
        tag = soup.find("meta", attrs={attr: key})
        if tag:
            val = str(tag.get("content", "")).strip()
            if val.startswith("http"):
                return val
    link = soup.find("link", rel="image_src")
    if link and str(link.get("href", "")).strip().startswith("http"):
        return str(link["href"]).strip()
    # 2. Amazon main image (handles data-a-dynamic-image, picks the largest).
    amazon = _image_from_amazon(soup)
    if amazon:
        return amazon
    # 3. Inline JSON blobs (AliExpress and similar).
    return _image_from_scripts(soup)


def _scrape_with_apify(url: str, apify_token: str) -> tuple[str | None, str | None]:
    """Returns (title, error_key). error_key is None on success."""
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
                        return cleaned, None
        return None, "apify_empty"
    except Exception as e:
        logger.warning(f"Apify scrape failed for {url}: {e}")
        return None, "apify_failed"


_APIFY_IMG_PAGEFUNC = (
    "async function pageFunction(context) {\n"
    "    await new Promise(r => setTimeout(r, 3000));\n"
    "    const q = (s) => document.querySelector(s);\n"
    "    let img = q('meta[property=\"og:image\"]')?.content\n"
    "        || q('meta[name=\"twitter:image\"]')?.content\n"
    "        || q('#landingImage')?.src\n"
    "        || q('link[rel=\"image_src\"]')?.href\n"
    "        || '';\n"
    "    if (!img) {\n"
    "        // No meta image: pick the largest rendered <img> (avoids logos/icons).\n"
    "        let best = '', bestArea = 0;\n"
    "        for (const im of document.querySelectorAll('img')) {\n"
    "            const src = im.currentSrc || im.src || '';\n"
    "            if (!src.startsWith('http')) continue;\n"
    "            const area = (im.naturalWidth || 0) * (im.naturalHeight || 0);\n"
    "            if (area > bestArea) { bestArea = area; best = src; }\n"
    "        }\n"
    "        img = best;\n"
    "    }\n"
    "    return { url: context.request.url, image: img };\n"
    "}"
)


def _scrape_images_apify_sync(urls: list[str], apify_token: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    try:
        from apify_client import ApifyClient
        client = ApifyClient(apify_token)
        run_input = {
            "startUrls": [{"url": u} for u in urls],
            "pageFunction": _APIFY_IMG_PAGEFUNC,
            "proxyConfiguration": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]},
            "maxConcurrency": 10,
            # Don't follow links — only the URLs we passed. Guards against a runaway
            # crawl inflating the Apify bill.
            "linkSelector": "",
            "maxPagesPerCrawl": len(urls) + 5,
        }
        run = client.actor("apify/web-scraper").call(run_input=run_input)
        logger.info(f"Apify image run finished, dataset_id={run.default_dataset_id}")
        for item in client.dataset(run.default_dataset_id).iterate_items():
            u = item.get("url")
            img = str(item.get("image") or "").strip()
            if u:
                out[u] = img if img.startswith("http") else None
    except Exception as e:
        logger.warning(f"Apify image batch failed: {e}")
    return out


async def scrape_images_apify(urls: list[str], apify_token: str) -> dict[str, str | None]:
    """Batch-resolve product images via ONE Apify run (residential proxy, renders JS).

    Returns {input_url: image_url | None}. Fallback for products whose image can't
    be scraped directly (IP-blocked or JS-only pages). 1688 is skipped (same as titles).
    """
    if not urls or not apify_token:
        return {}
    targets = [
        u for u in urls
        if not any(_get_domain(u) == d or _get_domain(u).endswith("." + d) for d in SKIP_APIFY_DOMAINS)
    ]
    if not targets:
        return {}
    return await asyncio.to_thread(_scrape_images_apify_sync, targets, apify_token)


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


async def scrape_product(url: str, timeout: float = 15.0) -> tuple[str | None, str | None, str | None]:
    """Single fetch → (title, image_url, error_key).

    Parses the page once and extracts BOTH title and image, preferring
    schema.org/Product JSON-LD, then og: tags, then CSS selectors, then the URL
    slug. error_key is None whenever a usable title was found.
    """
    host = _get_domain(url)
    is_1688 = host == "1688.com" or host.endswith(".1688.com")
    headers = HEADERS_1688 if is_1688 else HEADERS
    fetch_url = url.replace("detail.1688.com", "m.1688.com") if is_1688 else url

    soup, error_key = await _fetch_soup(fetch_url, timeout, headers)
    title = image = None
    if soup is not None:
        if _is_blocked(soup):
            error_key = "blocked"
        else:
            ld_name, ld_img = _extract_jsonld(soup)
            if ld_name:
                title = _clean_title(ld_name)
            if not title:
                title = _parse_title(soup)
            image = ld_img or _parse_image(soup)

    if not title:
        title = _title_from_url(url)

    return title, image, (None if title else error_key)


async def scrape_product_image_url(url: str, timeout: float = 10.0) -> str | None:
    """Extract main product image URL. Thin wrapper over scrape_product (used by CLIP build)."""
    _, image, _ = await scrape_product(url, timeout=timeout)
    return image


async def scrape_product_title_fast(url: str) -> tuple[str | None, str | None]:
    """Returns (title, error_key). Thin wrapper over scrape_product for back-compat."""
    title, _, error_key = await scrape_product(url)
    return title, error_key


async def scrape_product_title_apify(url: str, apify_token: str) -> tuple[str | None, str | None]:
    """Returns (title, error_key). error_key is None on success."""
    host = _get_domain(url)
    if any(host == d or host.endswith("." + d) for d in SKIP_APIFY_DOMAINS):
        return None, None
    return await asyncio.to_thread(_scrape_with_apify, url, apify_token)