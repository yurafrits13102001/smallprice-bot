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
# Lower gate: CLIP is only a coarse recall filter; GPT-4o vision is the final judge.
CLIP_RECALL = 0.55

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


def preload_model() -> None:
    """Eagerly load the CLIP model at startup so the first image query doesn't block."""
    _load_model()


def _embed_images(images: list) -> list[np.ndarray]:
    model, processor = _load_model()
    all_embeddings = []
    for i in range(0, len(images), 32):
        batch = images[i:i + 32]
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            pooled = model.vision_model(pixel_values=pixel_values).pooler_output
            features_np = model.visual_projection(pooled).detach().numpy()
        for feat in features_np:
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
        with httpx.Client(timeout=5, follow_redirects=True, headers=_HEADERS) as c:
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
        # product_idx -> image URL that was embedded (for GPT-4o vision compare)
        self.image_urls: dict[int, str] = {}

    def build(self, products, image_urls: list[str | None], chunk_size: int = 32) -> None:
        """Process images in chunks to avoid loading all into RAM at once."""
        tasks = [(i, url) for i, url in enumerate(image_urls) if url]
        if not tasks:
            logger.warning("CLIP build: no image URLs provided")
            return

        idx_to_url = dict(tasks)
        model, processor = _load_model()
        all_embeddings = []
        all_indices = []
        downloaded_total = 0

        for chunk_start in range(0, len(tasks), chunk_size):
            chunk = tasks[chunk_start:chunk_start + chunk_size]

            with ThreadPoolExecutor(max_workers=16) as ex:
                downloaded = list(ex.map(_download_image, chunk))

            valid = [(prod_idx, img) for prod_idx, img in downloaded if img is not None]
            downloaded_total += len(valid)
            if not valid:
                continue

            imgs = [img for _, img in valid]
            idxs = [i for i, _ in valid]

            inputs = processor(images=imgs, return_tensors="pt")
            pixel_values = inputs["pixel_values"]
            with torch.no_grad():
                pooled = model.vision_model(pixel_values=pixel_values).pooler_output
                features = model.visual_projection(pooled).detach().numpy()

            for prod_idx, feat in zip(idxs, features):
                feat = feat.astype(np.float32)
                norm = np.linalg.norm(feat)
                all_embeddings.append(feat / norm if norm > 0 else feat)
                all_indices.append(prod_idx)
                self.image_urls[prod_idx] = idx_to_url[prod_idx]

        logger.info(f"CLIP build: {downloaded_total}/{len(tasks)} images processed")
        if not all_embeddings:
            return

        matrix = np.array(all_embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(CLIP_DIM)
        self.index.add(matrix)
        self.product_indices = all_indices
        logger.info(f"CLIP index built: {self.index.ntotal} images for {len(products)} products")

    def search(self, image_url: str, products, top_k: int = 5) -> list[tuple]:
        """Returns list of (product, score, image_url). image_url may be '' for old indexes."""
        if not self.index or self.index.ntotal == 0:
            return []
        emb = get_image_embedding(image_url)
        if emb is None:
            return []
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(emb.reshape(1, -1), k)
        return [
            (products[self.product_indices[idx]], float(score), self.image_urls.get(self.product_indices[idx], ""))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def save(self, path: str) -> None:
        if not self.index:
            return
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump({
                "product_indices": self.product_indices,
                "image_urls": self.image_urls,
            }, f)
        logger.info(f"CLIP index saved: {self.index.ntotal} images → {path}")

    def load(self, path: str) -> bool:
        import os
        if not (os.path.exists(f"{path}.faiss") and os.path.exists(f"{path}.meta")):
            return False
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta", "rb") as f:
            meta = pickle.load(f)
        self.product_indices = meta["product_indices"]
        self.image_urls = meta.get("image_urls", {})
        logger.info(f"CLIP index loaded: {self.index.ntotal} images, {len(self.image_urls)} urls")
        return True
