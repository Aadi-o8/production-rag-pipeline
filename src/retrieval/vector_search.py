from typing import Optional
import chromadb
from chromadb.config import Settings

from src.utils.logger import get_logger
from src import config

logger = get_logger(__name__)


class VectorStore:

    def __init__(
        self,
        persist_dir: str = str(config.CHROMA_DIR),
        collection_name: str = config.CHROMA_COLLECTION_NAME,
    ):

        logger.info(f"Initializing ChromaDB at {persist_dir}")

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create the collection
        # If it already exists (from a previous run), ChromaDB loads it
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": config.CHROMA_DISTANCE_METRIC},
        )

        logger.info(
            f"Collection '{collection_name}' loaded: "
            f"{self.collection.count()} existing chunks"
        )

    def add_chunks(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:

        if not ids:
            return

        # ChromaDB handles deduplication by ID — if an ID exists, it updates
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Added/updated {len(ids)} chunks in vector store")

    def query(
        self,
        query_embedding: list[float],
        top_k: int = config.VECTOR_SEARCH_TOP_K,
        where: Optional[dict] = None,
    ) -> dict:

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            query_params["where"] = where

        results = self.collection.query(**query_params)

        # ChromaDB returns nested lists (for batch queries), unwrap the outer list
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }

    def get_all_documents(self) -> set[str]:

        if self.collection.count() == 0:
            return set()

        all_meta = self.collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in all_meta["metadatas"]}
        return sources

    def delete_by_source(self, source: str) -> int:

        # First, find all IDs with this source
        results = self.collection.get(
            where={"source": source},
            include=[],
        )
        ids = results["ids"]

        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks from source: {source}")

        return len(ids)

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        
        self.client.delete_collection(config.CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": config.CHROMA_DISTANCE_METRIC},
        )
        logger.warning("Vector store reset — all data deleted")