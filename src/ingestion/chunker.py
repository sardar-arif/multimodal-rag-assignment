from typing import List, Dict, Any
from src.core.logger import logger

class TextChunker:
    """
    Splits larger text blobs into overlapping chunks to ensure context window
    limits are respected and retrieval is focused.
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_page(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits a single page's text into overlapping chunks, preserving metadata.
        """
        content = page_data.get("content", "")
        if not content:
            return []

        # Simple whitespace-based tokenization approximation
        words = content.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            # No need to chunk if the page is short
            chunks.append(self._create_chunk_metadata(page_data, content))
            return chunks

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            chunks.append(self._create_chunk_metadata(page_data, chunk_text))
            
            # Move start forward, accounting for overlap
            start += (self.chunk_size - self.overlap)
            
            # Prevent infinite loops if overlap >= chunk_size
            if self.chunk_size <= self.overlap:
                break

        return chunks

    def _create_chunk_metadata(self, page_data: Dict[str, Any], chunk_text: str) -> Dict[str, Any]:
        """Utility to enforce the standard metadata schema for chunks."""
        return {
            "doc_id": page_data.get("doc_id", "unknown"),
            "page": page_data.get("page", -1),
            "chunk_type": "text",
            "content": chunk_text
        }
