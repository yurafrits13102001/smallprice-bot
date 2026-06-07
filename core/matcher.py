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
        self.client = AsyncOpenAI(api_key=api_key)
        self.index: faiss.IndexFlatIP | None = None
        self.products: list[Product] = []
        self.url_map: dict[str, int] = {}
        self.asin_map: dict[str, int] = {}
        self.aliexpress_map: dict[str, int] = {}
        self.clip_index = None

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

    async def build_clip_index_async(self, products: list[Product], save_path: str) -> None:
        from core.clip_matcher import CLIPImageIndex
        from core.scraper import scrape_product_image_url

        cache_path = Path(save_path).parent / "clip_image_cache.json"
        cache: dict[str, str | None] = {}
        try:
            cache = json.loads(cache_path.read_text())
            logger.info(f"CLIP cache loaded: {len(cache)} entries")
        except Exception:
            pass

        sem = asyncio.Semaphore(15)

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
            # Scrape only uncached URLs
            async with sem:
                results = await asyncio.gather(
                    *[scrape_product_image_url(u, timeout=4.0) for u in uncached],
                    return_exceptions=True,
                )
                for u, r in zip(uncached, results):
                    cache[u] = r if isinstance(r, str) and r else None
                for u, r in zip(uncached, results):
                    if isinstance(r, str) and r:
                        return r
            return None

        logger.info("CLIP: scraping product images...")
        img_urls = await asyncio.gather(*[_scrape_img(p) for p in products])
        found = sum(1 for u in img_urls if u)
        logger.info(f"CLIP: {found}/{len(products)} images found")

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

    async def search_by_image(self, image_url: str, top_k: int = 5) -> list[tuple[Product, float]]:
        if not self.clip_index:
            return []
        return await asyncio.to_thread(self.clip_index.search, image_url, self.products, top_k)

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