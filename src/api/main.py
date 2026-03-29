from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI(
    title="Multimodal RAG System - Tata EV Manuals",
    description="API for ingesting EV manuals and performing multimodal retrieval-augmented generation.",
    version="1.0.0"
)

class HealthStatus(BaseModel):
    status: str
    uptime_seconds: float
    timestamp: float

START_TIME = time.time()

@app.get("/health", response_model=HealthStatus)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
