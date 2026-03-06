import math
import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.char_recursive import _locate_chunks


def _split_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _merge_small_chunks(chunks: List[str], min_chars: int) -> List[str]:
    if not chunks:
        return []
    merged: List[str] = []
    buffer = ""
    for chunk in chunks:
        if not buffer:
            buffer = chunk
            continue
        if len(buffer) < min_chars:
            buffer = f"{buffer} {chunk}".strip()
        else:
            merged.append(buffer)
            buffer = chunk
    if buffer:
        merged.append(buffer)
    return merged


class SemanticChunker:
    name = "semantic"

    def __init__(
        self,
        similarity_threshold: float = 0.55,
        window_sentences: int = 4,
        min_chars: int = 600,
        max_chars: int = 2200,
        fallback_char_chunk_size: int = 1200,
        overlap_sentences: int = 1,
    ):
        self.similarity_threshold = similarity_threshold
        self.window_sentences = max(2, window_sentences)
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.fallback_char_chunk_size = fallback_char_chunk_size
        self.overlap_sentences = max(0, overlap_sentences)

    def _build_windows(self, sentences: List[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
        step = max(1, self.window_sentences - self.overlap_sentences)
        windows = []
        spans = []
        for idx in range(0, len(sentences), step):
            window = sentences[idx : idx + self.window_sentences]
            if not window:
                break
            windows.append(" ".join(window))
            spans.append((idx, min(idx + self.window_sentences, len(sentences))))
            if idx + self.window_sentences >= len(sentences):
                break
        return windows, spans

    def _compute_boundaries(
        self, sentences: List[str], window_spans: List[Tuple[int, int]], sims: List[float]
    ) -> List[int]:
        boundaries = [0]
        for idx, sim in enumerate(sims):
            if sim < self.similarity_threshold:
                cut = window_spans[idx][1]
                if cut not in boundaries:
                    boundaries.append(cut)
        boundaries.append(len(sentences))
        boundaries = sorted(set(boundaries))
        return boundaries

    def _split_large_chunk(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.fallback_char_chunk_size,
            chunk_overlap=max(200, int(self.fallback_char_chunk_size * 0.1)),
        )
        return splitter.split_text(text)

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Document]:
        sentences = _split_sentences(text)
        if len(sentences) <= self.window_sentences:
            from chunking.char_recursive import RecursiveCharChunker

            char_chunker = RecursiveCharChunker()
            return char_chunker.chunk(text, metadata=metadata)

        windows, spans = self._build_windows(sentences)
        if len(windows) <= 1:
            from chunking.char_recursive import RecursiveCharChunker

            char_chunker = RecursiveCharChunker()
            return char_chunker.chunk(text, metadata=metadata)

        from utility import _get_embedding_model

        model = _get_embedding_model()
        embeddings = model.embed_documents(windows)
        sims = [
            _cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]
        boundaries = self._compute_boundaries(sentences, spans, sims)

        raw_chunks = []
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            chunk_text = " ".join(sentences[start:end]).strip()
            if chunk_text:
                raw_chunks.append(chunk_text)

        refined_chunks = []
        for chunk in raw_chunks:
            if len(chunk) > self.max_chars:
                refined_chunks.extend(self._split_large_chunk(chunk))
            else:
                refined_chunks.append(chunk)
        refined_chunks = _merge_small_chunks(refined_chunks, self.min_chars)

        positions = _locate_chunks(text, refined_chunks)
        docs: List[Document] = []
        for idx, chunk_text in enumerate(refined_chunks):
            start_char, end_char = positions[idx]
            doc_meta = {
                "chunk_index": idx,
                "chunker_name": self.name,
                "start_char": start_char,
                "end_char": end_char,
                "level": "detail",
            }
            if metadata:
                doc_meta.update(metadata)
            docs.append(Document(page_content=chunk_text, metadata=doc_meta))
        return docs
