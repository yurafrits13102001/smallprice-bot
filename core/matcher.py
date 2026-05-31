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
        names = [p.name for p in products]

        logger.info(f"Building embeddings for {len(names)} products...")
        embeddings = await self._get_embeddings(names)

        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)

        self.url_map = {}
        for idx, product in enumerate(products):
            all_urls = [product.link] + product.supplier_links
            for url in all_urls:
                if url:
                    normalized = _normalize_url(url)
                    self.url_map[normalized] = idx

        logger.info(f"Index built: {self.index.ntotal} vectors, {len(self.url_map)} URLs")

    def save_index(self, path: str) -> None:
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump({"products": self.products, "url_map": self.url_map}, f)
        logger.info(f"Index saved to {path}")

    def load_index(self, path: str) -> None:
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta", "rb") as f:
            meta = pickle.load(f)
        self.products = meta["products"]
        self.url_map = meta["url_map"]
        logger.info(f"Index loaded: {self.index.ntotal} vectors, {len(self.url_map)} URLs")

    def _url_match(self, url: str) -> tuple[Product, float] | None:
        normalized = _normalize_url(url)
        idx = self.url_map.get(normalized)
        if idx is not None:
            return self.products[idx], 1.0
        return None

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