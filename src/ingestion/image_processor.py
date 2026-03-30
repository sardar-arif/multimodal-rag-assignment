import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.models.vlm import VLMModel
from src.core.logger import logger

class ImageProcessor:
    """
    Coordinates the processing of raw image files extracted from PDFs.
    Uses the VLM to generate textual summaries so they can be embedded
    and retrieved alongside standard text.
    """
    def __init__(self):
        self.vlm = VLMModel()

    def process_images(self, image_paths: List[str], base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes a list of image paths, translating each into a semantic text chunk.
        
        Args:
            image_paths: List of absolute or relative paths to image files.
            base_metadata: Dictionary containing doc_id, page number, etc.
            
        Returns:
            List of chunk dictionaries ready for embedding.
        """
        if not image_paths:
            logger.info("No images provided for processing.")
            return []

        semantic_chunks = []
        logger.info(f"Processing {len(image_paths)} images via VLM.")

        for image_path in image_paths:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found at path: {image_path}")
                continue

            summary = self.vlm.summarize_image(image_path)
            
            if summary:
                chunk_meta = base_metadata.copy()
                chunk_meta["chunk_type"] = "image"
                chunk_meta["content"] = summary
                
                # Keep original path just in case the UI needs to display it later
                chunk_meta["image_path"] = str(Path(image_path).absolute())
                
                semantic_chunks.append(chunk_meta)
            else:
                logger.warning(f"Skipping empty summary for image: {image_path}")

        logger.info(f"Successfully converted {len(semantic_chunks)} images to semantic chunks.")
        return semantic_chunks
