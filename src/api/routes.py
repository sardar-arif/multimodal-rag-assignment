from fastapi import APIRouter, Depends, UploadFile, File, BackgroundTasks, HTTPException
import time
import os
import uuid
from pathlib import Path
from src.api.schemas import QueryRequest, QueryResponse, IngestionResponse
from src.api.dependencies import get_retriever, get_generator, get_vector_store
from src.core.logger import logger
from src.ingestion.pipeline import IngestionPipeline

router = APIRouter()

# Directory for temporary file uploads
UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def background_ingest_task(file_path: str, vector_store, embedding_model):
    """
    Runs the full ingestion pipeline asynchronously so the API doesn't hang.
    """
    try:
        pipeline = IngestionPipeline(vector_store, embedding_model)
        result = pipeline.process_document(file_path)
        logger.info(f"Background ingestion completed for {file_path}. Metrics: {result}")
    except Exception as e:
        logger.error(f"Background ingestion failed for {file_path}: {str(e)}")
    finally:
        # Clean up the temporary uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Deleted temporary upload: {file_path}")

@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Uploads a PDF manual and kicks off the background multimodal ingestion pipeline.
    """
    logger.info(f"Received ingestion request for file: {file.filename}")
    
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Rejected non-PDF file upload: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save file locally with UUID to prevent naming collisions
    safe_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = str(UPLOAD_DIR / safe_filename)
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to save file for processing.")

    # We need the singletons to pass into the background task
    from src.api.dependencies import _vector_store, _embedding_model
    
    # Enqueue background processing
    background_tasks.add_task(background_ingest_task, file_path, _vector_store, _embedding_model)

    return IngestionResponse(
        status="processing",
        message="Document queued for background multimodal ingestion.",
        file_name=file.filename,
        task_id=str(uuid.uuid4())
    )

@router.post("/query", response_model=QueryResponse)
async def query_system(
    request: QueryRequest,
    retriever = Depends(get_retriever),
    generator = Depends(get_generator)
):
    """
    Takes a user query, retrieves multimodal chunks, and generates a grounded response.
    """
    logger.info(f"Received query request: '{request.query}' with top_k={request.top_k}")
    
    try:
        t0 = time.time()
        # 1. Retrieve Phase
        context_chunks = retriever.retrieve(request.query, top_k=request.top_k)
        t_retrieval = (time.time() - t0) * 1000
        
        if not context_chunks:
            logger.warning("No context chunks retrieved.")
        else:
            logger.debug(f"Retrieved {len(context_chunks)} context chunks.")

        t1 = time.time()
        # 2. Generation Phase
        result = generator.generate_answer(request.query, context_chunks)
        t_generation = (time.time() - t1) * 1000
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            retrieval_time_ms=round(t_retrieval, 2),
            generation_time_ms=round(t_generation, 2)
        )
    except Exception as e:
        logger.error(f"Failed to process query '{request.query}': {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during query processing.")
