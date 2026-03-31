from typing import List, Dict, Any
from src.retrieval.vector_store import VectorStore
from src.models.embeddings import EmbeddingModel
from src.core.logger import logger

class MultimodalRetriever:
    """
    Coordinates semantic search across the FAISS vector store.
    """
    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds a user query and searches the FAISS index.
        Returns the most relevant multimodal chunks with their metadata.
        """
        logger.info(f"Retrieving top {top_k} chunks for query: '{query}'")
        
        if self.vector_store.index.ntotal == 0:
            logger.warning("Retrieval attempted on empty vector store.")
            return []

        try:
            query_embedding = self.embedding_model.embed_query(query)
            results = self.vector_store.search(query_embedding, top_k=top_k)
            logger.info(f"Successfully retrieved {len(results)} chunks.")
            
            # Defensive check: ensure results actually carry valid content
            valid_results = [r for r in results if r.get("content")]
            
            if len(valid_results) < len(results):
                logger.warning(f"Filtered out {len(results) - len(valid_results)} chunks missing content.")
                
            return valid_results
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise
