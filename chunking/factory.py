from typing import Any

from chunking.base import BaseChunker
from chunking.char_recursive import RecursiveCharChunker
from chunking.semantic import SemanticChunker


def get_chunker(kind: str, **kwargs: Any) -> BaseChunker:
    kind = (kind or "char").lower()
    if kind == "semantic":
        return SemanticChunker(**kwargs)
    if kind == "char":
        return RecursiveCharChunker(
            chunk_size=kwargs.get("chunk_size"),
            overlap=kwargs.get("overlap"),
        )
    raise ValueError(f"Unknown chunker kind: {kind}")
