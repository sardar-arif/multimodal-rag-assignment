import google.generativeai as genai
from PIL import Image
from typing import Optional
from src.core.config import settings
from src.core.logger import logger

class VLMModel:
    """
    Wrapper for Google Gemini 1.5 Flash to act as a Vision Language Model.
    Summarizes images (dashboard icons, diagrams) extracted from EV manuals.
    """
    def __init__(self):
        logger.info(f"Initializing VLM using model: {settings.llm_model_name}")
        genai.configure(api_key=settings.google_api_key)
        self.model = genai.GenerativeModel(settings.llm_model_name)

    def summarize_image(self, image_path: str) -> Optional[str]:
        """
        Passes an image to Gemini to generate a dense, searchable text summary 
        contextualized for Tata EV Manuals.
        """
        try:
            img = Image.open(image_path)
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {str(e)}")
            return None

        prompt = (
            "You are an expert for Tata EV vehicles (like Nexon EV, Tiago EV). "
            "Examine this image extracted from an owner's manual. "
            "Describe the image in detail. If it is a dashboard warning icon, explain what it means. "
            "If it is a technical diagram, summarize its key components. "
            "Focus strictly on the factual, technical information presented in the image."
        )

        try:
            response = self.model.generate_content([prompt, img])
            summary = response.text.strip()
            logger.debug(f"Successfully generated summary for {image_path}")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary for image {image_path}: {str(e)}")
            return None
