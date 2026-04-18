from pathlib import Path

from src.ingestion.pipeline import ingest_file, ingest_directory
from src.ingestion.chunker import Chunk
from src.embeddings.embedder import Embedder
from src.retrieval.vector_search import VectorStore
from src.retrieval.keyword_search import KeywordSearch
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.generation.prompt_builder import build_prompt, build_sources_list
from src.generation.llm import LocalLLM
from src.utils.logger import get_logger
from src import config

logger = get_logger(__name__)


class RAGEngine:
    # Main RAG Engine class that orchestrates all components.
    def __init__(self):
        logger.info("=" * 60)
        logger.info("Initializing RAG Engine...")
        logger.info("=" * 60)

        # Initialize all components
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.keyword_search = KeywordSearch()
        self.reranker = Reranker()
        self.llm = LocalLLM()

        # Rebuild BM25 index from existing vector store data
        self._rebuild_bm25_index()

        logger.info("=" * 60)
        logger.info("RAG Engine ready!")
        logger.info(f"  Documents in store: {len(self.vector_store.get_all_documents())}")
        logger.info(f"  Total chunks: {self.vector_store.count()}")
        logger.info("=" * 60)

    def _rebuild_bm25_index(self) -> None:

        if self.vector_store.count() == 0:
            return

        # Get all documents and metadatas from ChromaDB
        all_data = self.vector_store.collection.get(
            include=["documents", "metadatas"]
        )

        if all_data["documents"]:
            self.keyword_search.index(
                documents=all_data["documents"],
                metadatas=all_data["metadatas"],
            )

    def ingest(self, path: Path) -> dict:

        logger.info(f"Starting ingestion: {path}")

        # Step 1: Parse and chunk
        if path.is_file():
            chunks = ingest_file(path)
        elif path.is_dir():
            chunks = ingest_directory(path)
        else:
            raise ValueError(f"Path does not exist: {path}")

        if not chunks:
            return {"files_processed": 0, "chunks_created": 0}

        # Step 2: Generate embeddings for all chunks
        chunk_texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(chunk_texts)} chunks...")
        embeddings = self.embedder.embed_texts(chunk_texts, show_progress=True)

        # Step 3: Store in ChromaDB
        ids = [f"{c.source}_chunk_{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
                **c.metadata,
            }
            for c in chunks
        ]

        self.vector_store.add_chunks(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunk_texts,
            metadatas=metadatas,
        )

        # Step 4: Rebuild BM25 index (now includes new documents)
        self._rebuild_bm25_index()

        stats = {
            "files_processed": len(set(c.source for c in chunks)),
            "chunks_created": len(chunks),
            "total_chunks_in_store": self.vector_store.count(),
        }

        logger.info(f"Ingestion complete: {stats}")
        return stats

    def query(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        source_filter: str | None = None,
    ) -> tuple[str, list[dict]]:

        if self.vector_store.count() == 0:
            return (
                "No documents have been ingested yet. Please upload some documents first.",
                [],
            )

        logger.info(f"Query: {question}")

        # Step 1: Embed the query
        query_embedding = self.embedder.embed_query(question)

        # Step 2: Vector search
        where_filter = {"source": source_filter} if source_filter else None
        vector_results = self.vector_store.query(
            query_embedding=query_embedding.tolist(),
            top_k=config.VECTOR_SEARCH_TOP_K,
            where=where_filter,
        )

        # Reshape vector results to match expected format
        vector_formatted = [
            {
                "document": doc,
                "metadata": meta,
                "distance": dist,
            }
            for doc, meta, dist in zip(
                vector_results["documents"],
                vector_results["metadatas"],
                vector_results["distances"],
            )
        ]

        # Step 3: Keyword search
        keyword_results = self.keyword_search.search(
            query=question,
            top_k=config.KEYWORD_SEARCH_TOP_K,
        )

        # Step 4: Hybrid merge
        hybrid_results = reciprocal_rank_fusion(
            vector_results=vector_formatted,
            keyword_results=keyword_results,
        )

        # Step 5: Re-rank
        reranked = self.reranker.rerank(
            query=question,
            results=hybrid_results,
            top_k=config.RERANK_TOP_K,
        )

        # Step 6: Build prompt
        prompt = build_prompt(
            query=question,
            retrieved_chunks=reranked,
            conversation_history=conversation_history,
        )

        # Step 7: Generate answer
        answer = self.llm.generate(prompt)

        # Step 8: Build source citations
        sources = build_sources_list(reranked)

        return answer, sources

    def get_ingested_documents(self) -> set[str]:
        """List all ingested document names."""
        return self.vector_store.get_all_documents()

    def delete_document(self, source: str) -> int:
        """Remove a document from the system."""
        count = self.vector_store.delete_by_source(source)
        self._rebuild_bm25_index()
        return count

    def reset(self) -> None:
        """Delete all data and reset the system."""
        self.vector_store.reset()
        self.keyword_search = KeywordSearch()
        logger.warning("RAG Engine reset — all data cleared")