from pathlib import Path

from src.ingestion.parsers import parse_document, DocumentResult
from src.ingestion.chunker import chunk_text, Chunk
from src.utils.logger import get_logger
from src import config

logger = get_logger(__name__)


def ingest_file(file_path: Path) -> list[Chunk]:

    doc_result: DocumentResult = parse_document(file_path)

    if not doc_result.text:
        logger.warning(f"No text from {file_path.name}, skipping")
        return []

    chunks = chunk_text(
        text=doc_result.text,
        source=doc_result.source,
        metadata=doc_result.metadata,
    )

    return chunks


def ingest_directory(dir_path: Path) -> list[Chunk]:

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    all_chunks = []
    files_processed = 0
    files_failed = 0

    supported_files = [
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
    ]

    if not supported_files:
        logger.warning(f"No supported files found in {dir_path}")
        return []

    logger.info(f"Found {len(supported_files)} files to ingest in {dir_path}")

    for file_path in sorted(supported_files):
        try:
            chunks = ingest_file(file_path)
            all_chunks.extend(chunks)
            files_processed += 1
            logger.info(f"  ✓ {file_path.name}: {len(chunks)} chunks")
        except Exception as e:
            files_failed += 1
            logger.error(f"  ✗ {file_path.name}: {e}")

    logger.info(
        f"Ingestion complete: {files_processed} files processed, "
        f"{files_failed} failed, {len(all_chunks)} total chunks"
    )

    return all_chunks