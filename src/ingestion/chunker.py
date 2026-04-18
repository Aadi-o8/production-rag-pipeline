from dataclasses import dataclass
from typing import Optional

import nltk

from src.utils.logger import get_logger
from src import config

logger = get_logger(__name__)


try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


@dataclass
class Chunk:

    text: str
    chunk_index: int
    source: str
    start_char: int
    end_char: int
    metadata: dict


def naive_chunk(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[tuple[str, int, int]]:

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append((chunk_text.strip(), start, min(end, len(text))))

        start += chunk_size - chunk_overlap

    return chunks


def sentence_aware_chunk(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[tuple[str, int, int]]:

    sentences = nltk.sent_tokenize(text)

    if not sentences:
        return []

    sentence_spans = []
    search_start = 0
    for sent in sentences:
        idx = text.find(sent, search_start)
        if idx == -1:
            # Fallback: if exact match fails (rare), use approximate position
            idx = search_start
        sentence_spans.append((sent, idx, idx + len(sent)))
        search_start = idx + len(sent)

    chunks = []
    current_sentences = []
    current_length = 0
    chunk_start_char = sentence_spans[0][1] if sentence_spans else 0

    for i, (sent, sent_start, sent_end) in enumerate(sentence_spans):
        sent_length = len(sent)

        # If adding this sentence exceeds chunk_size AND we already have content
        if current_length + sent_length > chunk_size and current_sentences:
            # Save current chunk
            chunk_text = " ".join(current_sentences)
            chunk_end_char = sentence_spans[i - 1][2]  # end of previous sentence
            chunks.append((chunk_text, chunk_start_char, chunk_end_char))

            # Calculate overlap: walk backwards through sentences until we've
            # accumulated enough characters for the overlap
            overlap_sentences = []
            overlap_length = 0
            for prev_sent in reversed(current_sentences):
                if overlap_length + len(prev_sent) > chunk_overlap:
                    break
                overlap_sentences.insert(0, prev_sent)
                overlap_length += len(prev_sent) + 1  # +1 for space

            # Start new chunk with overlap sentences
            current_sentences = overlap_sentences
            current_length = sum(len(s) + 1 for s in current_sentences)

            # Update start char to the beginning of the first overlap sentence
            if overlap_sentences:
                # Find the span of the first overlap sentence
                first_overlap = overlap_sentences[0]
                for s, s_start, s_end in sentence_spans:
                    if s == first_overlap and s_start >= chunk_start_char:
                        chunk_start_char = s_start
                        break
            # If no sentences fit in the overlap (e.g. very long sentence), we fit the previous sentence as the overlap forcibly
            if not overlap_sentences and current_sentences:
                overlap_sentences = [current_sentences[-1]]
            else:
                chunk_start_char = sent_start

        current_sentences.append(sent)
        current_length += sent_length + 1  # +1 for space between sentences

    # the last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk_end_char = sentence_spans[-1][2] if sentence_spans else len(text)
        chunks.append((chunk_text, chunk_start_char, chunk_end_char))

    return chunks


def chunk_text(
    text: str,
    source: str,
    metadata: Optional[dict] = None,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    use_semantic: bool = config.SEMANTIC_CHUNKING,
) -> list[Chunk]:

    if not text or not text.strip():
        logger.warning(f"Empty text received for {source}, returning no chunks")
        return []

    metadata = metadata or {}

    cleaned = "\n\n".join(
        " ".join(paragraph.split())
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    )

    if use_semantic:
        try:
            raw_chunks = sentence_aware_chunk(cleaned, chunk_size, chunk_overlap)
            logger.info(f"Sentence-aware chunking: {len(raw_chunks)} chunks from {source}")
        except Exception as e:
            logger.warning(f"Sentence chunking failed ({e}), falling back to naive")
            raw_chunks = naive_chunk(cleaned, chunk_size, chunk_overlap)
    else:
        raw_chunks = naive_chunk(cleaned, chunk_size, chunk_overlap)
        logger.info(f"Naive chunking: {len(raw_chunks)} chunks from {source}")

    chunks = []
    for i, (chunk_text, start_char, end_char) in enumerate(raw_chunks):
        if len(chunk_text) < config.MIN_CHUNK_SIZE:
            logger.debug(f"Skipping tiny chunk ({len(chunk_text)} chars) from {source}")
            continue

        chunks.append(Chunk(
            text=chunk_text,
            chunk_index=i,
            source=source,
            start_char=start_char,
            end_char=end_char,
            metadata={**metadata, "chunk_size": len(chunk_text)},
        ))

    logger.info(
        f"Chunked {source}: {len(cleaned)} chars → {len(chunks)} chunks "
        f"(avg {sum(len(c.text) for c in chunks) // max(len(chunks), 1)} chars/chunk)"
    )

    return chunks