import google.generativeai as genai
from typing import List, Dict, Any
from src.core.config import settings
from src.core.logger import logger

class RAGGenerator:
    """
    Manages the LLM prompt generation and completion using Google Gemini 1.5 Flash.
    Enforces strict grounding on the provided context retrieved from FAISS.
    """
    def __init__(self):
        logger.info(f"Initializing Generator LLM: {settings.llm_model_name}")
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(settings.llm_model_name)

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats multimodal chunks into a structured string for the LLM.
        """
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            ctype = chunk.get("chunk_type", "unknown")
            page = chunk.get("page", "unknown")
            content = chunk.get("content", "")
            
            context_parts.append(f"[Source {i+1} | Type: {ctype} | Page: {page}]\n{content}\n")
            
        return "\n".join(context_parts)

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a grounded answer based strictly on the retrieved chunks.
        """
        if not retrieved_chunks:
            logger.warning("No context provided to the LLM. Returning fallback answer.")
            return {
                "answer": "I do not have enough context in the owner manual to answer this question.",
                "sources": []
            }

        context_str = self.format_context(retrieved_chunks)
        
        prompt = f"""
You are an expert technical assistant for Tata electric vehicles (like Nexon EV, Tiago EV).
Your task is to answer the user's question using ONLY the provided context blocks below.

Rules:
1. Be precise and factual.
2. If the context does not contain the answer, explicitly state that you don't know based on the manual. Do NOT guess.
3. You must include a "Citations" section at the bottom of your answer referencing the [Source X] blocks.

Context:
{context_str}

Question:
{query}

Answer:
"""
        try:
            logger.info("Generating final answer using LLM...")
            response = self.model.generate_content(prompt)
            answer_text = response.text.strip()
            
            # Extract basic citation metadata from chunks for the API response payload
            sources = []
            for chunk in retrieved_chunks:
                sources.append({
                    "page": chunk.get("page"),
                    "type": chunk.get("chunk_type"),
                    "doc_id": chunk.get("doc_id")
                })
                
            return {
                "answer": answer_text,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Failed to generate LLM completion: {str(e)}")
            raise
