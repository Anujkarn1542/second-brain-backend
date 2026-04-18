import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.routers import ingest, query

load_dotenv()

app = FastAPI(
    title="Second Brain API",
    description="RAG-powered document Q&A backend",
    version="1.0.0",
)

# CORS — allows your React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/")
def root():
    return {"status": "ok", "message": "Second Brain API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}