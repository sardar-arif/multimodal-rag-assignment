import time
from typing import Dict, Any, List
from pathlib import Path
from src.core.logger import logger
from src.ingestion.parser import PDFParser
from src.ingestion.chunker import TextChunker
from src.ingestion.table_processor import TableProcessor
from src.ingestion.image_processor import ImageProcessor
from src.models.embeddings import EmbeddingModel
from src.retrieval.vector_store import VectorStore

class IngestionPipeline:
    """
    Orchestrates the end-to-end flow from raw PDF to searchable FAISS index.
    Coordinates specialized processors for text, tables, and images.
    """
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        
        # Initialize processors
        self.chunker = TextChunker()
        self.table_processor = TableProcessor()
        self.image_processor = ImageProcessor()

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Runs the full ingestion pipeline on a single document.
        """
        start_time = time.time()
        logger.info(f"Starting ingestion pipeline for: {file_path}")
        
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Cannot ingest missing document: {file_path}")

        # 1. Parse Document (Text extraction implemented, tables/images mocked for now)
        parser = PDFParser(file_path)
        pages_data = parser.extract_text_pages()
        
        all_chunks = []

        # 2. Process Text
        text_chunks_count = 0
        for page in pages_data:
            chunks = self.chunker.chunk_page(page)
            all_chunks.extend(chunks)
            text_chunks_count += len(chunks)

        # 3. Process Tables (Placeholder for actual table extraction logic)
        # In a full system, `parser` would also return `extracted_tables`
        extracted_tables = [] # Replace with actual extraction 
        table_chunks_count = 0
        for table, metadata in extracted_tables:
            chunks = self.table_processor.process_table(table, metadata)
            all_chunks.extend(chunks)
            table_chunks_count += len(chunks)

        # 4. Process Images (Placeholder for actual image extraction logic)
        # In a full system, `parser` would save images to disk and return paths
        extracted_image_paths = [] # Replace with actual paths
        image_metadata = {"doc_id": path_obj.name, "page": 1} # Base metadata
        image_chunks = self.image_processor.process_images(extracted_image_paths, image_metadata)
        all_chunks.extend(image_chunks)
        image_chunks_count = len(image_chunks)

        # 5. Embed and Store
        if all_chunks:
            logger.info("Generating embeddings for all extracted multimodal chunks...")
            texts_to_embed = [chunk["content"] for chunk in all_chunks]
            embeddings = self.embedding_model.embed_texts(texts_to_embed)
            
            logger.info("Adding mapped embeddings and metadata to Vector Store...")
            self.vector_store.add_chunks(embeddings, all_chunks)
            self.vector_store.save()
        else:
            logger.warning("No chunks were generated from this document.")

        processing_time = time.time() - start_time
        
        # 6. Return standard metrics payload
        result = {
            "file": path_obj.name,
            "text_chunks": text_chunks_count,
            "table_chunks": table_chunks_count,
            "image_chunks": image_chunks_count,
            "total_chunks": len(all_chunks),
            "processing_time_seconds": round(processing_time, 2)
        }
        
        logger.info(f"Pipeline complete for {path_obj.name} in {result['processing_time_seconds']}s")
        return result
