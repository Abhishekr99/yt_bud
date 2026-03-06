from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _select_chunk_params(text_length: int) -> Tuple[int, int]:
    if text_length > 300000:
        return 4000, 400
    if text_length > 150000:
        return 6000, 600
    if text_length > 80000:
        return 8000, 800
    return 10000, 1000


def _locate_chunks(text: str, chunks: List[str]) -> List[Tuple[int, int]]:
    cursor = 0
    positions: List[Tuple[int, int]] = []
    for chunk in chunks:
        start = text.find(chunk, cursor)
        if start == -1:
            start = cursor
        end = start + len(chunk)
        positions.append((start, end))
        cursor = end
    return positions


class RecursiveCharChunker:
    name = "char"

    def __init__(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Document]:
        size, overlap = _select_chunk_params(len(text))
        chunk_size = self.chunk_size or size
        chunk_overlap = self.overlap if self.overlap is not None else overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        positions = _locate_chunks(text, chunks)
        docs: List[Document] = []
        for idx, chunk_text in enumerate(chunks):
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
