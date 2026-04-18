from pathlib import Path
import os

# ─── Document Parsing ────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE_MB = 50  # reject files larger than this

# ─── Project Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "documents"
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
EMBEDDING_CACHE_DIR = STORAGE_DIR / "cache" / "embeddings"
LLM_CACHE_DIR = STORAGE_DIR / "cache" / "llm"

# ─── Chunking ────────────────────────────────────────────────────
CHUNK_SIZE = 1000          # characters (not tokens)
CHUNK_OVERLAP = 200        # ~20% overlap
MIN_CHUNK_SIZE = 100       # discard tiny leftover chunks
SEMANTIC_CHUNKING = True   # use sentence-boundary-aware chunking

# ─── Embedding Model ─────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim embeddings, ~80MB model, fast on CPU/GPU
# It maps sentences to a 384-dimensional dense vector space.
# Trained on 1B+ sentence pairs — great for semantic similarity.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
EMBEDDING_BATCH_SIZE = 64  # how many chunks to embed at once
USE_GPU_FOR_EMBEDDINGS = True  # using my RTX 4060

# ─── Vector DB (ChromaDB) ────────────────────────────────────────
CHROMA_COLLECTION_NAME = "rag_documents"
CHROMA_DISTANCE_METRIC = "cosine"

# ─── Retrieval ────────────────────────────────────────────────────
VECTOR_SEARCH_TOP_K = 20       # initial broad retrieval
KEYWORD_SEARCH_TOP_K = 20      # BM25 results
HYBRID_TOP_K = 20              # after merging vector + keyword
RERANK_TOP_K = 5               # final results after re-ranking

# Hybrid search weight: 0.0 = pure keyword, 1.0 = pure vector
# 0.7 gives more weight to semantic search but keyword still contributes
HYBRID_VECTOR_WEIGHT = 0.7

# ─── Re-ranker ────────────────────────────────────────────────────
# Cross-encoder model: takes (query, document) pairs and scores relevance.
# Much more accurate than bi-encoder (embedding similarity) but slower,
# which is why we only re-rank the top candidates.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ─── LLM (Local - Phi-3-mini) ────────────────────────────────────
LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
LLM_MAX_NEW_TOKENS = 512      # max tokens in LLM response
LLM_TEMPERATURE = 0.1         # low temp = more focused/factual
LLM_CONTEXT_WINDOW = 4096     # Phi-3-mini supports 4k tokens
# Reserve tokens for the prompt template + question
LLM_CONTEXT_BUDGET = 3000     # max tokens for retrieved context

# ─── Caching ──────────────────────────────────────────────────────
ENABLE_EMBEDDING_CACHE = True
ENABLE_LLM_CACHE = True
LLM_CACHE_MAX_SIZE = 1000     # max cached responses

# ─── Streamlit UI ─────────────────────────────────────────────────
MAX_UPLOAD_FILES = 10
MAX_CONVERSATION_HISTORY = 50