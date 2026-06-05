import io
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor

import faiss
import httpx
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_DIM = 512
CLIP_THRESHOLD = 0.75
CLIP_CONFIDENT = 0.85

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

_model: CLIPModel | None = None
_processor: CLIPProcessor | None = None


def _load_model():
    global _model, _processor
    if _model is None:
        logger.info("Loading CLIP model...")
        _model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _model.eval()
        logger.info("CLIP model loaded")
    return _model, _processor


def _embed_images(images: list) -> list[np.ndarray]:
    model, processor = _load_model()
    all_embeddings = []
    for i in range(0, len(images), 8):
        batch = images[i:i + 8]
        inputs = processor(images=batch, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs).numpy()
        for feat in features:
            feat = feat.astype(np.float32)
            norm = np.linalg.norm(feat)
            all_embeddings.append(feat / norm if norm > 0 else feat)
    return all_embeddings


def get_image_embedding(image_url: str) -> np.ndarray | None:
    try:
        with httpx.Client(timeout=10, follow_redirects=True, headers=_HEADERS) as c:
            r = c.get(image_url)
            if r.status_code != 200:
                return None
            image = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        logger.debug(f"Image fetch failed {image_url[:60]}: {e}")
        return None
    result = _embed_images([image])
    return result[0] if result else None


def _download_image(args: tuple) -> tuple[int, object | None]:
    prod_idx, url = args
    try:
        with httpx.Client(timeout=8, follow_redirects=True, headers=_HEADERS) as c:
            r = c.get(url)
            if r.status_code == 200:
                return prod_idx, Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        pass
    return prod_idx, None


class CLIPImageIndex:
    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.product_indices: list[int] = []

    def build(self, products, image_urls: list[str | None]) -> None:
        tasks = [(i, url) for i, url in enumerate(image_urls) if url]
        if not tasks:
            logger.warning("CLIP build: no image URLs provided")
            return

        with ThreadPoolExecutor(max_workers=8) as ex:
            downloaded = list(ex.map(_download_image, tasks))

        valid = [(prod_idx, img) for prod_idx, img in downloaded if img is not None]
        logger.info(f"CLIP build: {len(valid)}/{len(tasks)} images downloaded")
        if not valid:
            return

        prod_indices = [i for i, _ in valid]
        images = [img for _, img in valid]
        embeddings = _embed_images(images)

        matrix = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(CLIP_DIM)
        self.index.add(matrix)
        self.product_indices = prod_indices
        logger.info(f"CLIP index built: {self.index.ntotal} images for {len(products)} products")

    def search(self, image_url: str, products, top_k: int = 5) -> list[tuple]:
        if not self.index or self.index.ntotal == 0:
            return []
        emb = get_image_embedding(image_url)
        if emb is None:
            return []
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(emb.reshape(1, -1), k)
        return [
            (products[self.product_indices[idx]], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def save(self, path: str) -> None:
        if not self.index:
            return
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump({"product_indices": self.product_indices}, f)
        logger.info(f"CLIP index saved: {self.index.ntotal} images → {path}")

    def load(self, path: str) -> bool:
        import os
        if not (os.path.exists(f"{path}.faiss") and os.path.exists(f"{path}.meta")):
            return False
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta", "rb") as f:
            meta = pickle.load(f)
        self.product_indices = meta["product_indices"]
        logger.info(f"CLIP index loaded: {self.index.ntotal} images")
        return True
