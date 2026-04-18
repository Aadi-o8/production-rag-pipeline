from sentence_transformers import CrossEncoder

from src.utils.logger import get_logger
from src import config

logger = get_logger(__name__)


class Reranker:

    def __init__(self, model_name: str = config.RERANKER_MODEL_NAME):
        logger.info(f"Loading re-ranker model: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info("Re-ranker ready")

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = config.RERANK_TOP_K,
    ) -> list[dict]:

        if not results:
            return []

        pairs = [(query, r["document"]) for r in results]

        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

        logger.info(
            f"Re-ranked {len(results)} → {len(reranked)} results. "
            f"Score range: {min(scores):.3f} to {max(scores):.3f}"
        )

        return reranked