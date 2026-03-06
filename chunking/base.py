from typing import Protocol

from langchain_core.documents import Document


class BaseChunker(Protocol):
    name: str

    def chunk(self, text: str, *, metadata: dict | None = None) -> list[Document]:
        ...
