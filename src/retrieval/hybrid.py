from src.utils.logger import get_logger
from src import config

logger = get_logger(__name__)

# RRF constant — empirically 60 works well across many benchmarks
RRF_K = 60


def reciprocal_rank_fusion(
    vector_results: list[dict],
    keyword_results: list[dict],
    vector_weight: float = config.HYBRID_VECTOR_WEIGHT,
    top_k: int = config.HYBRID_TOP_K,
) -> list[dict]:

    keyword_weight = 1.0 - vector_weight

    doc_scores: dict[str, dict] = {}

    # Process vector results
    for rank, result in enumerate(vector_results):
        doc_text = result["document"]
        rrf_contribution = vector_weight * (1.0 / (RRF_K + rank + 1))

        if doc_text not in doc_scores:
            doc_scores[doc_text] = {
                "document": doc_text,
                "metadata": result["metadata"],
                "rrf_score": 0.0,
                "vector_rank": rank + 1,
                "keyword_rank": None,
                "vector_distance": result.get("distance"),
            }

        doc_scores[doc_text]["rrf_score"] += rrf_contribution

    # Process keyword results
    for rank, result in enumerate(keyword_results):
        doc_text = result["document"]
        rrf_contribution = keyword_weight * (1.0 / (RRF_K + rank + 1))

        if doc_text not in doc_scores:
            doc_scores[doc_text] = {
                "document": doc_text,
                "metadata": result["metadata"],
                "rrf_score": 0.0,
                "vector_rank": None,
                "keyword_rank": rank + 1,
                "bm25_score": result.get("score"),
            }

        doc_scores[doc_text]["rrf_score"] += rrf_contribution
        doc_scores[doc_text]["keyword_rank"] = rank + 1

    sorted_results = sorted(
        doc_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )[:top_k]

    both_count = sum(
        1 for r in sorted_results
        if r.get("vector_rank") is not None and r.get("keyword_rank") is not None
    )
    logger.info(
        f"Hybrid search: {len(vector_results)} vector + {len(keyword_results)} keyword "
        f"→ {len(sorted_results)} merged ({both_count} appeared in both)"
    )

    return sorted_results