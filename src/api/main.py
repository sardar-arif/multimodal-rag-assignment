from fastapi import FastAPI
from pydantic import BaseModel
import time
from src.api.routes import router

app = FastAPI(
    title="Multimodal RAG System - Tata EV Manuals",
    description="API for ingesting EV manuals and performing multimodal retrieval-augmented generation.",
    version="1.0.0"
)

# Include the main routes for /ingest and /query
app.include_router(router)

from fastapi import Depends
from src.api.dependencies import get_vector_store
from src.retrieval.vector_store import VectorStore

class HealthStatus(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    index_dimension: int
    uptime_seconds: float
    timestamp: float

START_TIME = time.time()

@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check(vector_store: VectorStore = Depends(get_vector_store)):
    """
    Verifies the server is operational and returns system status,
    including the active count of FAISS indexed documents and isolated chunks.
    """
    unique_docs = set([meta.get("doc_id") for meta in vector_store.metadata_store.values() if "doc_id" in meta])
    
    return HealthStatus(
        status="healthy",
        total_documents=len(unique_docs),
        total_chunks=vector_store.index.ntotal,
        index_dimension=vector_store.dimension,
        uptime_seconds=round(time.time() - START_TIME, 2),
        timestamp=time.time()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
