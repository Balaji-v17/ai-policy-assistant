from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=key)

print("--- AVAILABLE MODELS IN 2026 ---")
try:
    # This asks Google for every model your key can use
    for m in client.models.list():
        if 'generateContent' in m.supported_actions:
            print(f"-> {m.name}")
except Exception as e:
    print(f"ERROR: {str(e)}")
