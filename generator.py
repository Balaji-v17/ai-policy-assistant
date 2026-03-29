from google import genai
import os
import logging
from dotenv import load_dotenv

# Load the .env file if it exists
load_dotenv()

class AIGenerator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.error("CRITICAL: GEMINI_API_KEY is missing!")
        
        # Initialize the 2026 Client
        self.client = genai.Client(api_key=api_key)
        # We use 1.5-flash because it is the most stable free model
        self.model_id = 'gemini-3-flash-preview' 

    async def is_available(self):
        return True

    async def generate_response(self, query: str, context_docs: list):
        try:
            context_text = "\n\n".join(context_docs)
            
            prompt = f"""
            You are a Professional Policy Assistant. 
            Use ONLY the provided context to answer the question.
            
            CONTEXT:
            {context_text}

            QUESTION:
            {query}
            """
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            return {"answer": response.text}
        except Exception as e:
            # This prints the REAL error to your terminal so we can see it
            logging.error(f"Gemini API Error: {str(e)}")
            return {"answer": "I'm having trouble connecting to the cloud brain."}
