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

class HealthStatus(BaseModel):
    status: str
    uptime_seconds: float
    timestamp: float

START_TIME = time.time()

@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """
    Verifies the server is operational and returns system status.
    """
    return HealthStatus(
        status="operational",
        uptime_seconds=time.time() - START_TIME,
        timestamp=time.time()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
