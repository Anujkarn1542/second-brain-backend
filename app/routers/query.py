import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.services.retriever import (
    search_chunks,
    generate_answer,
    generate_answer_stream,
    get_sources_from_chunks,
)
from app.models.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    chunks = search_chunks(
        question=request.question,
        document_id=request.document_id,
        top_k=5,
    )
    answer, sources = generate_answer(request.question, chunks)
    return QueryResponse(answer=answer, sources=sources)


@router.post("/stream")
async def query_documents_stream(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = search_chunks(
        question=request.question,
        document_id=request.document_id,
        top_k=5,
    )
    sources = get_sources_from_chunks(chunks)

    def event_generator():
        try:
            for text_piece in generate_answer_stream(request.question, chunks):
                data = json.dumps({"type": "chunk", "content": text_piece})
                yield f"data: {data}\n\n"
        except Exception:
            error_data = json.dumps(
                {
                    "type": "chunk",
                    "content": "Error generating response. Please try again.",
                }
            )
            yield f"data: {error_data}\n\n"

        # Serialize sources with score explicitly as float
        sources_payload = [
            {
                "document_name": s.document_name,
                "page_number": s.page_number,
                "snippet": s.snippet,
                "score": round(float(s.score), 3),
            }
            for s in sources
        ]
        sources_data = json.dumps(
            {
                "type": "sources",
                "sources": sources_payload,
            }
        )
        yield f"data: {sources_data}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
