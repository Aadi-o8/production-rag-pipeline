import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils.logger import get_logger
from src import config

logger = get_logger(__name__)


class EmbeddingCache:
    #Simple disk-based cache for text embeddings.
    def __init__(self, cache_dir: Path = config.EMBEDDING_CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_path(self, text_hash: str) -> Path:
        subdir = self.cache_dir / text_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{text_hash}.json"

    def get(self, text: str) -> Optional[list[float]]:
        text_hash = self._hash_text(text)
        cache_path = self._cache_path(text_hash)

        if cache_path.exists():
            self.hits += 1
            data = json.loads(cache_path.read_text())
            return data["embedding"]

        self.misses += 1
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        text_hash = self._hash_text(text)
        cache_path = self._cache_path(text_hash)
        cache_path.write_text(json.dumps({
            "text_hash": text_hash,
            "embedding": embedding,
        }))

    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {"hits": self.hits, "misses": self.misses, "hit_rate": f"{hit_rate:.1%}"}


class Embedder:

    def __init__(
        self,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        use_gpu: bool = config.USE_GPU_FOR_EMBEDDINGS,
        use_cache: bool = config.ENABLE_EMBEDDING_CACHE,
    ):
        from sentence_transformers import SentenceTransformer
        import torch

        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        logger.info(f"Loading embedding model: {model_name} (device={self.device})")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.cache = EmbeddingCache() if use_cache else None

        logger.info(
            f"Embedder ready: {config.EMBEDDING_DIMENSION}-dim vectors, "
            f"device={self.device}, cache={'on' if use_cache else 'off'}"
        )

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:

        if not texts:
            return np.array([])

        # Check cache for each text
        embeddings = [None] * len(texts)
        texts_to_embed = []  # (original_index, text) for cache misses
        cache_hit_count = 0

        if self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    embeddings[i] = cached
                    cache_hit_count += 1
                else:
                    texts_to_embed.append((i, text))
        else:
            texts_to_embed = list(enumerate(texts))

        if cache_hit_count > 0:
            logger.info(f"Embedding cache: {cache_hit_count}/{len(texts)} hits")

        # Embed the cache misses
        if texts_to_embed:
            miss_texts = [t for _, t in texts_to_embed]
            new_embeddings = self.model.encode(
                miss_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
            )

            # Fill in results and update cache
            for j, (orig_idx, text) in enumerate(texts_to_embed):
                emb = new_embeddings[j].tolist()
                embeddings[orig_idx] = emb
                if self.cache:
                    self.cache.put(text, emb)

        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0]