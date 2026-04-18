from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    document_id: str
    name: str
    page_count: int
    chunk_count: int
    message: str


class QueryRequest(BaseModel):
    question: str
    document_id: Optional[str] = None  # None = search across all docs


class Source(BaseModel):
    document_name: str
    page_number: int
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class ErrorResponse(BaseModel):
    message: str