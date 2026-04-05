from typing import Any
from src.retrieval.vector_store import VectorStore
from src.models.embeddings import EmbeddingModel
from src.retrieval.retriever import MultimodalRetriever
from src.models.llm import RAGGenerator

# Initialize singletons to prevent model reloading on every API call
_embedding_model = EmbeddingModel()
_vector_store = VectorStore(_embedding_model.dimension)
_retriever = MultimodalRetriever(_vector_store, _embedding_model)
_generator = RAGGenerator()

def get_retriever() -> MultimodalRetriever:
    return _retriever

def get_generator() -> RAGGenerator:
    return _generator

def get_vector_store() -> VectorStore:
    return _vector_store
