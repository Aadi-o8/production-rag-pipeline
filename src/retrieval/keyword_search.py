import re

from rank_bm25 import BM25Okapi

from src.utils.logger import get_logger
from src import config

logger = get_logger(__name__)


def tokenize(text: str) -> list[str]:

    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if len(t) >= 2]


class KeywordSearch:

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.tokenized_corpus: list[list[str]] = []
        self._indexed = False

    def index(self, documents: list[str], metadatas: list[dict]) -> None:

        if not documents:
            logger.warning("No documents to index for BM25")
            return

        self.documents = documents
        self.metadatas = metadatas

        self.tokenized_corpus = [tokenize(doc) for doc in documents]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._indexed = True

        logger.info(f"BM25 index built: {len(documents)} documents")

    def search(
        self,
        query: str,
        top_k: int = config.KEYWORD_SEARCH_TOP_K,
    ) -> list[dict]:

        if not self._indexed or self.bm25 is None:
            logger.warning("BM25 index not built, returning empty results")
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(scores[idx]),
                    "rank": rank,
                })

        logger.info(f"BM25 search: '{query}' → {len(results)} results")
        return results