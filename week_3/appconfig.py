import os
from dotenv import load_dotenv

load_dotenv()


class AppConfig:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Default model configurations
        self.default_llm_type = os.getenv("DEFAULT_LLM_TYPE", "openai")
        self.default_llm_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self.default_embedding_type = os.getenv("DEFAULT_EMBEDDING_TYPE", "openai")
        self.default_temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0"))
        self.default_whisper_model = os.getenv("DEFAULT_WHISPER_MODEL", "base")


app_config = AppConfig()

