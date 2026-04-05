from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class QueryRequest(BaseModel):
    query: str = Field(..., example="What does the turtle icon on the dashboard mean?")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")

class SourceCitation(BaseModel):
    page: int
    type: str
    doc_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    retrieval_time_ms: float
    generation_time_ms: float

class IngestionResponse(BaseModel):
    status: str
    message: str
    file_name: str
    task_id: str
