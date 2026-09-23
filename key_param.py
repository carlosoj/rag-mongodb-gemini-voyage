import os

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

def _get_required_env(var_name: str) -> str:
    """Retrieve an environment variable or raise an error if missing."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: '{var_name}'. "
            f"Ensure it is defined in your environment or .env file."
        )
    return value

# Database & API Credentials
MONGODB_URI: str = _get_required_env("MONGODB_URI")
VOYAGE_API_KEY: str = _get_required_env("VOYAGE_API_KEY")
GEMINI_API_KEY: str = _get_required_env("GEMINI_API_KEY")
SOURCE_FILE_PATH: str = _get_required_env("SOURCE_BOOK_PATH")

# Optional Credentials
GPT_API_KEY: str | None = os.getenv("LLM_API_KEY")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")