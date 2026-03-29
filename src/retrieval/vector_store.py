import faiss
import numpy as np
import json
import uuid
from typing import List, Dict, Any, Tuple
from pathlib import Path
from src.core.config import settings
from src.core.logger import logger

class VectorStore:
    """
    Manages the FAISS index for fast similarity search and a local JSON
    store to map vector indices back to their rich metadata (citations, source text).
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index_path = Path(settings.vector_db_path)
        self.metadata_path = Path(settings.metadata_db_path)
        
        # Ensure directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.index = self._load_or_create_index()
        self.metadata_store = self._load_or_create_metadata()

    def _load_or_create_index(self) -> faiss.IndexFlatL2:
        if self.index_path.exists():
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            return faiss.read_index(str(self.index_path))
        else:
            logger.info(f"Creating new FAISS index with dimension {self.dimension}")
            return faiss.IndexFlatL2(self.dimension)

    def _load_or_create_metadata(self) -> Dict[str, Any]:
        if self.metadata_path.exists():
            logger.info("Loading existing metadata store.")
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.info("Initializing new metadata store.")
            return {}

    def save(self):
        """Persists the index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata_store, f, indent=2)
        logger.info("Vector DB and metadata successfully saved to disk.")

    def add_chunks(self, embeddings: List[List[float]], metadata_list: List[Dict[str, Any]]):
        """
        Adds new text chunks and their embeddings to the store.
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Mismatched counts between embeddings and metadata.")
            
        if not embeddings:
            logger.warning("No embeddings provided to add.")
            return

        # FAISS requires float32 numpy arrays
        vectors_np = np.array(embeddings).astype('float32')
        
        # We need to map FAISS sequential IDs back to our metadata.
        # FAISS assigns IDs sequentially starting from its current ntotal.
        start_idx = self.index.ntotal
        
        self.index.add(vectors_np)
        
        # Store metadata mapping (string keys for JSON compatibility)
        for i, meta in enumerate(metadata_list):
            faiss_id = str(start_idx + i)
            # Add a unique generated ID for internal tracking just in case
            meta["chunk_id"] = str(uuid.uuid4())
            self.metadata_store[faiss_id] = meta
            
        logger.info(f"Added {len(embeddings)} new vectors. Total in index: {self.index.ntotal}")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches the index for the most similar chunks.
        """
        if self.index.ntotal == 0:
            logger.warning("Search attempted on empty index.")
            return []

        # Convert query to format required by FAISS
        query_np = np.array([query_embedding]).astype('float32')
        
        # distances (L2) and indices of the nearest neighbors
        distances, indices = self.index.search(query_np, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: # FAISS returns -1 if it can't find enough vectors
                continue
                
            str_idx = str(idx)
            if str_idx in self.metadata_store:
                meta = self.metadata_store[str_idx].copy()
                meta["distance"] = float(dist)
                results.append(meta)
                
        return results
