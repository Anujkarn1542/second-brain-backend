from pydantic import BaseModel, Field
from typing import Optional


class UploadResponse(BaseModel):
    document_id: str
    name: str
    page_count: int
    chunk_count: int
    message: str


class QueryRequest(BaseModel):
    question: str
    document_id: Optional[str] = None


class Source(BaseModel):
    document_name: str
    page_number: int
    snippet: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)  # always present, always 0-1


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class ErrorResponse(BaseModel):
    message: str
