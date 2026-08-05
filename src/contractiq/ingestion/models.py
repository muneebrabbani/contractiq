from __future__ import annotations

from pydantic import BaseModel


class Page(BaseModel):
    page_number: int
    text: str
    ocr: bool = False


class SourceDocument(BaseModel):
    doc_id: str
    file_path: str
    file_format: str
    pages: list[Page]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    page_number: int
    text: str
    token_count: int
