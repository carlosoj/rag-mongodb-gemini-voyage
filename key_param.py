import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SOURCE_FILE_PATH = os.getenv("SOURCE_BOOK_PATH")
HF_TOKEN = os.getenv("HF_TOKEN")