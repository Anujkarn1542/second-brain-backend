# import os
# import chromadb
# from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
# from dotenv import load_dotenv

# load_dotenv()

# CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# gemini_ef = GoogleGenerativeAiEmbeddingFunction(
#     api_key=GEMINI_API_KEY,
#     model_name="models/embedding-001"
# )

# chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


# def get_or_create_collection(document_id: str):
#     return chroma_client.get_or_create_collection(
#         name=f"doc_{document_id}",
#         embedding_function=gemini_ef,
#         metadata={"hnsw:space": "cosine"}
#     )


# def store_chunks(document_id: str, chunks: list[dict]) -> int:
#     collection = get_or_create_collection(document_id)

#     ids = [chunk["id"] for chunk in chunks]
#     texts = [chunk["text"] for chunk in chunks]
#     metadatas = [
#         {
#             "page_number": chunk["page_number"],
#             "document_name": chunk["document_name"],
#         }
#         for chunk in chunks
#     ]

#     collection.add(ids=ids, documents=texts, metadatas=metadatas)
#     return len(chunks)


# def delete_document(document_id: str):
#     try:
#         chroma_client.delete_collection(f"doc_{document_id}")
#     except Exception:
#         pass

import os
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_or_create_collection(document_id: str):
    return chroma_client.get_or_create_collection(
        name=f"doc_{document_id}"
    )


def store_chunks(document_id: str, chunks: list[dict]) -> int:
    collection = get_or_create_collection(document_id)

    ids = [chunk["id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "page_number": chunk["page_number"],
            "document_name": chunk["document_name"],
        }
        for chunk in chunks
    ]

    # 🚫 NO embedding function used here
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )

    return len(chunks)


def delete_document(document_id: str):
    try:
        chroma_client.delete_collection(f"doc_{document_id}")
    except Exception:
        pass