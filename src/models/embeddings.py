from sentence_transformers import SentenceTransformer
from typing import List, Union
from src.core.config import settings
from src.core.logger import logger

class EmbeddingModel:
    """
    Wrapper for local embedding generation using SentenceTransformers.
    Provides completely free, private vectorization.
    """
    def __init__(self):
        logger.info(f"Initializing embedding model: {settings.embedding_model_name}")
        # Loads locally; internet needed only on first boot to download weights
        self.model = SentenceTransformer(settings.embedding_model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of string.
        """
        try:
            # encode returns a numpy array, we convert to lists for easier handling
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding for a single query string.
        """
        try:
            embedding = self.model.encode(query, show_progress_bar=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise
