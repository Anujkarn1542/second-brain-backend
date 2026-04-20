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
# model = genai.GenerativeModel("gemini-1.5-flash")


def search_chunks(question: str, document_id: str | None, top_k: int = 5) -> list[dict]:
    try:
        if document_id:
            collections = [get_or_create_collection(document_id)]
        else:
            all_cols = chroma_client.list_collections()
            if not all_cols:
                return []

            collections = []
            for col in all_cols:
                try:
                    collections.append(chroma_client.get_collection(col.name))
                except Exception:
                    continue

    except Exception:
        return []

    all_results = []

    for collection in collections:
        try:
            count = collection.count()
            if count == 0:
                continue

            results = collection.query(
                query_texts=[question],
                n_results=min(top_k, count),
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
                    "score": round(1 - distance, 3),
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
            # Clamp score between 0 and 1, default 0 if missing
            raw_score = chunk.get("score", 0.0)
            safe_score = max(0.0, min(1.0, float(raw_score)))
            sources.append(
                Source(
                    document_name=chunk["document_name"],
                    page_number=chunk["page_number"],
                    snippet=chunk["text"][:150] + "...",
                    score=safe_score,
                )
            )
    return sources


def generate_answer(question: str, chunks: list[dict]) -> tuple[str, list[Source]]:
    if not chunks:
        return ("I couldn't find relevant information in your documents.", [])

    try:
        context = build_context(chunks)

        prompt = f"""
You are an AI assistant that answers questions based ONLY on the provided document.

IMPORTANT RULES:
- If user asks for summarize, explain, or list points, help using context
- Use only context
- If no answer exists, say clearly
- Be concise and helpful

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        response = model.generate_content(prompt)

        return response.text or "No response generated.", get_sources_from_chunks(chunks)

    except Exception:
        return (
            "Something went wrong while generating the answer. Please try again.",
            []
        )


def generate_answer_stream(
    question: str, chunks: list[dict]
) -> Generator[str, None, None]:
    if not chunks:
        yield "I couldn't find relevant information in your documents."
        return

    context = build_context(chunks)
    prompt = f"""You are a helpful assistant answering questions strictly based on provided document context.

Rules:
- Only use information from the context below
- If context doesn't contain the answer say "I couldn't find this in the uploaded documents"
- Be concise and direct
- Never make up information

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    # except Exception as e:
    #     print("GEMINI ERROR:", repr(e))
    #     yield f"Gemini Error: {str(e)}"
    except Exception:
        yield "Something went wrong generating the response. Please try again."
