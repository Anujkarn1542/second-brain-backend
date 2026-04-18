import os
import uuid
import traceback
from fastapi import APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from app.services.chunker import (
    extract_text_from_pdf,
    extract_text_from_txt,
    chunk_pages,
)
from app.services.embedder import store_chunks, delete_document
from app.models.schemas import UploadResponse

load_dotenv()

router = APIRouter(prefix="/ingest", tags=["ingest"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()

    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    document_id = str(uuid.uuid4())
    file_extension = ".pdf" if file.content_type == "application/pdf" else ".txt"
    file_path = os.path.join(UPLOAD_DIR, f"{document_id}{file_extension}")

    with open(file_path, "wb") as f:
        f.write(content)

    try:
        if file.content_type == "application/pdf":
            pages = extract_text_from_pdf(file_path)
        else:
            pages = extract_text_from_txt(file_path)

        if not pages:
            raise HTTPException(
                status_code=422,
                detail="Could not extract text. Is it a scanned PDF?"
            )

        original_name = file.filename or "document"
        chunks = chunk_pages(pages, original_name)
        chunk_count = store_chunks(document_id, chunks)

        return UploadResponse(
            document_id=document_id,
            name=original_name,
            page_count=len(pages),
            chunk_count=chunk_count,
            message="Document processed successfully."
        )

    except HTTPException:
        raise
    except Exception as e:
        # Print full traceback so we can see exact error
        traceback.print_exc()
        if os.path.exists(file_path):
            os.remove(file_path)
        delete_document(document_id)
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/stats")
async def get_stats():
    try:
        from app.services.embedder import chroma_client
        collections = chroma_client.list_collections()

        docs = []
        total_chunks = 0

        for col in collections:
            collection = chroma_client.get_collection(col.name)
            count = collection.count()
            total_chunks += count

            # Get metadata from first item to extract document name
            if count > 0:
                results = collection.get(limit=1, include=["metadatas"])
                name = "Unknown"
                if results["metadatas"]:
                    name = results["metadatas"][0].get("document_name", "Unknown")

                docs.append({
                    "id": col.name.replace("doc_", ""),
                    "name": name,
                    "chunk_count": count,
                })

        return {
            "total_documents": len(collections),
            "total_chunks": total_chunks,
            "total_queries": 0,   # extend later with a DB
            "documents": docs,
        }
    except Exception:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0,
            "documents": [],
        }