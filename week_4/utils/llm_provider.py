from langchain_openai import ChatOpenAI


class LLMProvider:
    """Provider for OpenAI LLM"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """Initialize OpenAI LLM"""
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
    
    def get_llm(self) -> ChatOpenAI:
        """Get or create LLM instance"""
        if self.llm is None:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=self.api_key
            )
        return self.llm

