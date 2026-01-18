# constants.py
import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

class AppConfig:
    _instance = None

    def __init__(self):
        # Defaults
        self.OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
        self.TIKTOKEN_ENCODING = os.getenv("TIKTOKEN_ENCODING", "cl100k_base") # Default for GPT-4/3.5
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", 8000))
        # Auto-detect API type from base URL
        self._detect_api_type()

    def _detect_api_type(self):
        """Detect whether we're using v1/chat/completions or v1/responses API."""
        if "/v1/responses" in self.OPENAI_BASE_URL:
            self.API_TYPE = "v1_responses"
        elif "/v1/chat/completions" in self.OPENAI_BASE_URL:
            self.API_TYPE = "v1_chat_completions"
        else:
            # Default to chat completions for backward compatibility
            self.API_TYPE = "v1_chat_completions"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def update(self, base_url=None, api_key=None):
        """Runtime update for builder pattern"""
        if base_url:
            self.OPENAI_BASE_URL = base_url
            self._detect_api_type()  # Re-detect API type when base URL changes
        if api_key:
            self.OPENAI_API_KEY = api_key

config = AppConfig.get_instance()