import asyncio
import json
import logging
import pickle
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import faiss
from openai import AsyncOpenAI

from core.database import Product

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
BATCH_SIZE = 100


_KNOWN_DOMAINS = {
    "aliexpress.com",
    "amazon.com",
    "temu.com",
    "1688.com",
    "ebay.com",
}


def _normalize_url(url: str) -> str:
    try:
        parsed = urlparse(url.strip().rstrip("/"))
        host = parsed.netloc.lower()
        host = host.removeprefix("www.")
        for base in _KNOWN_DOMAINS:
            if host == base or host.endswith("." + base):
                host = base
                break
        path = parsed.path.rstrip("/")
        return f"{host}{path}"
    except Exception:
        return url.strip().lower()


def _extract_asin(url: str) -> str | None:
    m = re.search(r'/(?:dp|gp/product)/([A-Z0-9]{10})\b', url)
    return m.group(1) if m else None


def _extract_aliexpress_item_id(url: str) -> str | None:
    m = re.search(r'/(?:item|i)/(\d{10,})', url)
    return m.group(1) if m else None


def _extract_url_description(url: str) -> str | None:
    """Extract product description from URL slug (works for Amazon and similar)."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        for part in reversed(parts):
            if len(part) < 10 or part.isdigit():
                continue
            if "-" in part or "_" in part:
                candidate = re.sub(r"[-_]+", " ", part).strip()
                if re.search(r"[a-zA-Z]{3,}", candidate):
                    return candidate
    except Exception:
        pass
    return None


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9\-]{2,}", text.lower())
    stop_words = {
        "the", "and", "for", "with", "new", "hot", "set", "kit",
        "pcs", "pack", "style", "type", "size", "color", "from",
        "high", "quality", "free", "shipping", "sale", "best",
        "pro", "max", "mini", "plus", "ultra", "super", "original",
    }
    return {w for w in words if w not in stop_words and len(w) > 1}


def _keyword_boost(query: str, product_name: str) -> float:
    query_kw = _extract_keywords(query)
    product_kw = _extract_keywords(product_name)
    if not query_kw or not product_kw:
        return 0.0
    intersection = query_kw & product_kw
    if not intersection:
        return 0.0
    ratio = len(intersection) / min(len(query_kw), len(product_kw))
    return ratio * 0.15


class ProductMatcher:
    def __init__(self, api_key: str):
        # Shared client reused across every request (embeddings, normalize, verify,
        # vision) instead of a fresh one per message. max_retries adds SDK-level
        # backoff on transient 429/5xx/timeout for all of them.
        self.client = AsyncOpenAI(api_key=api_key, max_retries=3)
        self.index: faiss.IndexFlatIP | None = None
        self.products: list[Product] = []
        self.url_map: dict[str, int] = {}
        self.asin_map: dict[str, int] = {}
        self.aliexpress_map: dict[str, int] = {}
        self.clip_index = None
        self._index_path: str | None = None
        self._img_cache: dict | None = None

    async def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            response = await self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Embeddings: {i + len(batch)}/{len(texts)}")
        return np.array(all_embeddings, dtype=np.float32)

    async def build_index(self, products: list[Product]) -> None:
        self.products = products

        texts = []
        for p in products:
            desc = _extract_url_description(p.link)
            if not desc:
                for sup in p.supplier_links:
                    desc = _extract_url_description(sup)
                    if desc:
                        break
            texts.append(f"{p.name} {desc}" if desc else p.name)

        logger.info(f"Building embeddings for {len(texts)} products...")
        embeddings = await self._get_embeddings(texts)

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)

        self.url_map = {}
        self.asin_map = {}
        self.aliexpress_map = {}
        for idx, product in enumerate(products):
            all_urls = [product.link] + product.supplier_links
            for url in all_urls:
                if not url:
                    continue
                self.url_map[_normalize_url(url)] = idx
                asin = _extract_asin(url)
                if asin and asin not in self.asin_map:
                    self.asin_map[asin] = idx
                item_id = _extract_aliexpress_item_id(url)
                if item_id and item_id not in self.aliexpress_map:
                    self.aliexpress_map[item_id] = idx

        logger.info(f"Index built: {self.index.ntotal} vectors, {len(self.url_map)} URLs, {len(self.asin_map)} ASINs, {len(self.aliexpress_map)} AliExpress IDs")

    def save_index(self, path: str) -> None:
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump({
                "products": self.products,
                "url_map": self.url_map,
                "asin_map": self.asin_map,
                "aliexpress_map": self.aliexpress_map,
            }, f)
        if self.clip_index:
            self.clip_index.save(f"{path}_clip")
        logger.info(f"Index saved to {path}")

    def load_index(self, path: str) -> None:
        self._index_path = path
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta", "rb") as f:
            meta = pickle.load(f)
        self.products = meta["products"]
        self.url_map = meta["url_map"]
        self.asin_map = meta.get("asin_map", {})
        self.aliexpress_map = meta.get("aliexpress_map", {})
        logger.info(f"Index loaded: {self.index.ntotal} vectors, {len(self.url_map)} URLs, {len(self.asin_map)} ASINs, {len(self.aliexpress_map)} AliExpress IDs")
        from core.clip_matcher import CLIPImageIndex
        self.clip_index = CLIPImageIndex()
        if not self.clip_index.load(f"{path}_clip"):
            self.clip_index = None
            logger.info("CLIP index not found, image search disabled until next index rebuild")

    def _url_match(self, url: str) -> tuple[Product, float] | None:
        normalized = _normalize_url(url)
        idx = self.url_map.get(normalized)
        if idx is not None:
            return self.products[idx], 1.0

        asin = _extract_asin(url)
        if asin:
            idx = self.asin_map.get(asin)
            if idx is not None:
                logger.info(f"ASIN match: {asin}")
                return self.products[idx], 1.0

        item_id = _extract_aliexpress_item_id(url)
        if item_id:
            idx = self.aliexpress_map.get(item_id)
            if idx is not None:
                logger.info(f"AliExpress ID match: {item_id}")
                return self.products[idx], 1.0

        return None

    def _is_definitive_no_match(self, url: str) -> bool:
        asin = _extract_asin(url)
        if asin is not None:
            return asin not in self.asin_map
        item_id = _extract_aliexpress_item_id(url)
        if item_id is not None:
            return item_id not in self.aliexpress_map
        return False

    async def build_clip_index_async(
        self,
        products: list[Product],
        save_path: str,
        apify_token: str = "",
        *,
        concurrency: int = 6,
        scrape_timeout: float = 15.0,
        use_playwright: bool = False,
        playwright_concurrency: int = 3,
        playwright_proxy: str = "",
    ) -> None:
        from core.clip_matcher import CLIPImageIndex
        from core.scraper import scrape_product_image_url

        cache_path = Path(save_path).parent / "clip_image_cache.json"
        cache: dict[str, str | None] = {}
        try:
            cache = json.loads(cache_path.read_text())
            logger.info(f"CLIP cache loaded: {len(cache)} entries")
        except Exception:
            pass

        # Throttle-aware scraping. The semaphore caps how many PRODUCTS scrape at
        # once, and within a product we now try URLs one-by-one and stop at the
        # first hit — so peak load is ~`concurrency` requests, not concurrency×4.
        # The old code fired every URL of every in-flight product in parallel
        # (~60 simultaneous requests), which AliExpress throttled hard: a probe
        # got 23% sequentially but the burst build got 0.5%. Gentler + slower
        # per product, but far higher yield. Tunable for marketplaces that need it.
        sem = asyncio.Semaphore(concurrency)

        async def _scrape_img(p: Product) -> str | None:
            urls = list(dict.fromkeys(u for u in [p.link] + p.supplier_links if u))
            # Return first positive cached result
            for u in urls:
                if cache.get(u):
                    return cache[u]
            # Skip if all URLs already tried and failed
            uncached = [u for u in urls if u not in cache]
            if not uncached:
                return None
            # Try uncached URLs sequentially, stopping at the first image. This
            # both cuts the request burst and avoids scraping a product's other
            # URLs once one already yielded an image.
            async with sem:
                for u in uncached:
                    try:
                        r = await scrape_product_image_url(u, timeout=scrape_timeout)
                    except Exception:
                        r = None
                    cache[u] = r if isinstance(r, str) and r else None
                    if cache[u]:
                        return cache[u]
            return None

        logger.info("CLIP: scraping product images...")
        img_urls = list(await asyncio.gather(*[_scrape_img(p) for p in products]))
        found = sum(1 for u in img_urls if u)
        logger.info(f"CLIP: {found}/{len(products)} images found (direct)")

        # Headless-browser fallback (Playwright): recovers JS-rendered images the
        # static fetch can't see — 1688 (all of it) and most AliExpress. Runs on
        # our host, so no per-call cost; it's the default fallback now that free
        # Apify credit is exhausted. Tries EVERY URL of each still-missing product;
        # results cached under a "pw::" key so rebuilds don't re-render attempted URLs.
        if use_playwright:
            from core.browser_scraper import scrape_images_playwright
            pw_prod_urls: dict[int, list[str]] = {}
            pw_to_fetch: list[str] = []
            seen: set[str] = set()
            for i, p in enumerate(products):
                if img_urls[i]:
                    continue
                urls = list(dict.fromkeys(u for u in [p.link] + p.supplier_links if u))
                if not urls:
                    continue
                pw_prod_urls[i] = urls
                for u in urls:
                    if f"pw::{u}" not in cache and u not in seen:
                        seen.add(u)
                        pw_to_fetch.append(u)
            if pw_to_fetch:
                logger.info(f"CLIP: Playwright fallback — {len(pw_prod_urls)} product(s), {len(pw_to_fetch)} URL(s)...")
                # Render in chunks and flush the cache after each — a browser pass
                # over thousands of URLs takes hours, so this keeps it resumable:
                # a re-run skips every URL already recorded under a "pw::" key.
                CHUNK = 200
                for start in range(0, len(pw_to_fetch), CHUNK):
                    batch = pw_to_fetch[start:start + CHUNK]
                    pw_results = await scrape_images_playwright(
                        batch,
                        concurrency=playwright_concurrency,
                        proxy=playwright_proxy,
                    )
                    for u in batch:
                        cache[f"pw::{u}"] = pw_results.get(u)
                    try:
                        cache_path.write_text(json.dumps(cache))
                    except Exception as e:
                        logger.warning(f"CLIP cache save failed mid-Playwright: {e}")
                    done = min(start + CHUNK, len(pw_to_fetch))
                    logger.info(f"CLIP: Playwright {done}/{len(pw_to_fetch)} URLs processed")
            for i, urls in pw_prod_urls.items():
                if img_urls[i]:
                    continue
                hit = next((cache[f"pw::{u}"] for u in urls if cache.get(f"pw::{u}")), None)
                if hit:
                    img_urls[i] = hit
            if pw_prod_urls:
                recovered = sum(1 for i in pw_prod_urls if img_urls[i])
                found = sum(1 for u in img_urls if u)
                logger.info(f"CLIP: Playwright recovered {recovered}/{len(pw_prod_urls)}; {found}/{len(products)} images total")

        # Apify fallback: recover images for products the direct scrape missed
        # (IP-blocked or JS-only pages). Tries EVERY URL of each still-missing
        # product (not just the primary link) so a blocked primary can still be
        # recovered via a supplier link. ONE batched run; results cached under an
        # "apify::" key so rebuilds don't re-run Apify on already-attempted URLs.
        if apify_token:
            from core.scraper import scrape_images_apify
            prod_urls: dict[int, list[str]] = {}
            to_fetch: set[str] = set()
            for i, p in enumerate(products):
                if img_urls[i]:
                    continue
                urls = list(dict.fromkeys(u for u in [p.link] + p.supplier_links if u))
                if not urls:
                    continue
                prod_urls[i] = urls
                for u in urls:
                    if f"apify::{u}" not in cache:
                        to_fetch.add(u)
            if to_fetch:
                logger.info(f"CLIP: Apify image fallback — {len(prod_urls)} product(s), {len(to_fetch)} URL(s)...")
                apify_results = await scrape_images_apify(list(to_fetch), apify_token)
                for u in to_fetch:
                    cache[f"apify::{u}"] = apify_results.get(u)
            # Resolve each missing product from the (now updated) apify cache — uses
            # the first URL that yielded an image, including previously cached hits.
            for i, urls in prod_urls.items():
                hit = next((cache[f"apify::{u}"] for u in urls if cache.get(f"apify::{u}")), None)
                if hit:
                    img_urls[i] = hit
            if prod_urls:
                recovered = sum(1 for i in prod_urls if img_urls[i])
                found = sum(1 for u in img_urls if u)
                logger.info(f"CLIP: Apify recovered {recovered}/{len(prod_urls)}; {found}/{len(products)} images total")

        # Save updated cache
        try:
            cache_path.write_text(json.dumps(cache))
            logger.info(f"CLIP cache saved: {len(cache)} entries")
        except Exception as e:
            logger.warning(f"CLIP cache save failed: {e}")

        clip = CLIPImageIndex()
        await asyncio.to_thread(clip.build, products, list(img_urls))
        self.clip_index = clip
        if clip.index:
            clip.save(f"{save_path}_clip")
            logger.info("CLIP index saved")

    def _cached_image_url(self, product: Product) -> str:
        """Look up a product's scraped image URL from the CLIP build cache (fallback for old indexes)."""
        if self._img_cache is None:
            self._img_cache = {}
            try:
                cache_path = Path(self._index_path).parent / "clip_image_cache.json" if self._index_path else None
                if cache_path and cache_path.exists():
                    self._img_cache = json.loads(cache_path.read_text())
            except Exception:
                self._img_cache = {}
        for url in [product.link] + product.supplier_links:
            if url and self._img_cache.get(url):
                return self._img_cache[url]
        return ""

    async def search_by_image(self, image_url: str, top_k: int = 5) -> list[tuple[Product, float, str]]:
        """Returns (product, score, candidate_image_url). image_url filled from cache if index lacks it."""
        if not self.clip_index:
            return []
        results = await asyncio.to_thread(self.clip_index.search, image_url, self.products, top_k)
        enriched = []
        for product, score, img in results:
            if not img:
                img = self._cached_image_url(product)
            enriched.append((product, score, img))
        return enriched

    async def search(self, query: str, url: str = "", top_k: int = 5) -> list[tuple[Product, float]]:
        if url:
            url_result = self._url_match(url)
            if url_result:
                return [url_result]

        query_embedding = await self._get_embeddings([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k * 3)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            base_score = float(score)
            boost = _keyword_boost(query, self.products[idx].name)
            final_score = min(base_score + boost, 1.0)
            results.append((self.products[idx], final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]