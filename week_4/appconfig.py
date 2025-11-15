import os
from dotenv import load_dotenv

load_dotenv()


class AppConfig:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Default model configurations
        self.default_llm_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
        self.default_temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))


app_config = AppConfig()

