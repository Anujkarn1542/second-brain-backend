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
    """
    Streams the answer as Server-Sent Events (SSE).
    Frontend reads this as a stream and appends words one by one.

    SSE format:
      data: {"type": "chunk", "content": "Hello"}
      data: {"type": "sources", "sources": [...]}
      data: {"type": "done"}
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = search_chunks(
        question=request.question,
        document_id=request.document_id,
        top_k=5,
    )
    sources = get_sources_from_chunks(chunks)

    def event_generator():
        # Stream text chunks
        for text_piece in generate_answer_stream(request.question, chunks):
            data = json.dumps({"type": "chunk", "content": text_piece})
            yield f"data: {data}\n\n"

        # Send sources after text is done
        sources_data = json.dumps({
            "type": "sources",
            "sources": [s.model_dump() for s in sources]
        })
        yield f"data: {sources_data}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # important for nginx
        },
    )