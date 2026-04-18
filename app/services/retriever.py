import os
from typing import Generator
import google.generativeai as genai
from dotenv import load_dotenv
from app.services.embedder import get_or_create_collection, chroma_client
from app.models.schemas import Source

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash-lite")


def search_chunks(question: str, document_id: str | None, top_k: int = 5) -> list[dict]:
    if document_id:
        collections = [get_or_create_collection(document_id)]
    else:
        all_cols = chroma_client.list_collections()
        collections = [chroma_client.get_collection(col.name) for col in all_cols]

    all_results = []

    for collection in collections:
        try:
            results = collection.query(
                query_texts=[question],
                n_results=min(top_k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            if not results["documents"] or not results["documents"][0]:
                continue
            for text, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                all_results.append({
                    "text": text,
                    "page_number": metadata.get("page_number", 1),
                    "document_name": metadata.get("document_name", "Unknown"),
                    "score": 1 - distance,
                })
        except Exception:
            continue

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}: {chunk['document_name']}, Page {chunk['page_number']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def get_sources_from_chunks(chunks: list[dict]) -> list[Source]:
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["document_name"], chunk["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append(Source(
                document_name=chunk["document_name"],
                page_number=chunk["page_number"],
                snippet=chunk["text"][:150] + "...",
            ))
    return sources


def generate_answer(question: str, chunks: list[dict]) -> tuple[str, list[Source]]:
    if not chunks:
        return (
            "I couldn't find relevant information in your documents.",
            []
        )
    context = build_context(chunks)
    prompt = f"""
You are an AI assistant that answers questions based ONLY on the provided document.

IMPORTANT RULES:
- If user asks for "all text", "summarize", or "list questions", try to infer from context
- If partial information exists, still answer using available content
- Do NOT say "not found" unless absolutely nothing relevant exists
- Be helpful and interpret user intent

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = model.generate_content(prompt)
    return response.text or "", get_sources_from_chunks(chunks)


def generate_answer_stream(question: str, chunks: list[dict]) -> Generator[str, None, None]:
    """
    Streams the answer token by token using Gemini's stream mode.
    Yields raw text chunks as they arrive.
    """
    if not chunks:
        yield "I couldn't find relevant information in your documents."
        return

    context = build_context(chunks)
    prompt = f"""You are a helpful assistant that answers questions strictly based on the provided document context.

Rules:
- Only use information from the context below
- If the context doesn't contain the answer, say "I couldn't find this in the uploaded documents"
- Be concise and direct
- Do not make up information

Context:
{context}

Question: {question}

Answer:"""

    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text