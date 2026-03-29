from typing import List, Dict, Any, Optional
from src.core.logger import logger

class TableProcessor:
    """
    Handles the conversion of unstructured or structured table data into
    semantically meaningful sentences so they can be securely embedded
    and retrieved by the vector store.
    """
    def __init__(self):
        logger.info("Initializing Semantic Table Processor.")

    def process_table(self, table_data: List[List[str]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Converts a 2D list of table rows (where row 0 is usually headers) 
        into a list of chunk dictionaries containing semantic sentences.
        
        Args:
            table_data: A list of lists representing table rows.
            metadata: Base metadata dictionary (doc_id, page, etc.)
            
        Returns:
            List of chunk dictionaries ready for embedding.
        """
        if not table_data or len(table_data) < 2:
            logger.warning("Table data is empty or lacks sufficient rows for processing.")
            return []

        headers = [str(header).strip() for header in table_data[0]]
        semantic_chunks = []

        # Iterate through data rows (skip header row)
        for row_idx, row in enumerate(table_data[1:]):
            sentence_parts = []
            
            # Defensive check: ensure row matches header length
            valid_length = min(len(headers), len(row))
            
            for col_idx in range(valid_length):
                col_name = headers[col_idx]
                cell_value = str(row[col_idx]).strip()
                
                # Skip empty cells to maintain natural sentence flow
                if cell_value and cell_value.lower() not in ["none", "n/a", "null", "-"]:
                    sentence_parts.append(f"the {col_name} is {cell_value}")

            if sentence_parts:
                # E.g. "For this item, the Charging Mode is Fast, and the Time is 60 min."
                semantic_sentence = "For this table entry, " + ", and ".join(sentence_parts) + "."
                
                # Create standard chunk metadata
                chunk_meta = metadata.copy()
                chunk_meta["chunk_type"] = "table"
                chunk_meta["content"] = semantic_sentence
                chunk_meta["table_row_index"] = row_idx + 1 # 1-indexed for logging
                
                semantic_chunks.append(chunk_meta)

        logger.info(f"Successfully processed table into {len(semantic_chunks)} semantic sentences.")
        return semantic_chunks

    def parse_html_table(self, html_content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Placeholder for parsing raw HTML tables if extracted via layout analysis tools.
        In a production environment, this would utilize BeautifulSoup or pandas.read_html.
        """
        # For phase 3 foundation, we define the interface. 
        # Actual HTML parsing logic can be injected here when layout parsers are integrated.
        logger.warning("HTML table parsing is not yet implemented. Returning empty list.")
        return []
