import fitz  # PyMuPDF
from typing import List, Dict, Any
from pathlib import Path
from src.core.logger import logger

class PDFParser:
    """
    Handles extraction of text from PDF documents using PyMuPDF.
    """
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found at {file_path}")

    def extract_text_pages(self) -> List[Dict[str, Any]]:
        """
        Extracts raw text from each page.
        Returns a list of dictionaries containing page metadata and content.
        """
        logger.info(f"Extracting text from {self.file_path.name}")
        pages_data = []
        try:
            doc = fitz.open(self.file_path)
            for page_num, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    pages_data.append({
                        "doc_id": self.file_path.name,
                        "page": page_num + 1,  # 1-indexed for citations
                        "content": text
                    })
            doc.close()
            logger.info(f"Successfully extracted {len(pages_data)} pages with text.")
            return pages_data
        except Exception as e:
            logger.error(f"Failed to parse PDF {self.file_path.name}: {str(e)}")
            raise
