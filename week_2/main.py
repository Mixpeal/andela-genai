from groq import Groq
from openai import OpenAI
from anthropic import Anthropic
from appconfig import app_config
import time

PROVIDER_MODELS = {
  "groq": [
    "llama-3.3-70b-versatile",
    "llama-3.3-70b-chat",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
  ],
  "openai": [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o",
    "gpt-4o-mini",
  ],
  "anthropic": [
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
  ]
}

class LLMApp: 
  def __init__(self, api_key, model_name, provider="groq"):
    self.provider = provider
    
    if provider == "groq":
      self.api_key = api_key or app_config.groq_api_key
      if not self.api_key:
        raise ValueError("GROQ_API_KEY is not set")
      self.client = Groq(api_key=self.api_key)
    elif provider == "openai":
      self.api_key = api_key or app_config.openai_api_key
      if not self.api_key:
        raise ValueError("OPENAI_API_KEY is not set")
      self.client = OpenAI(api_key=self.api_key)
    elif provider == "anthropic":
      self.api_key = api_key or app_config.anthropic_api_key
      if not self.api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set")
      self.client = Anthropic(api_key=self.api_key)
    else:
      raise ValueError(f"Unsupported provider: {provider}")
    
    self.model = model_name
    self.conversation_history = []
    
  def chat(self, user_message, system_prompt=None, temperature=0.5, max_tokens=1024):
    system_prompt = system_prompt if system_prompt else "You are a helpful assistant named Andelina, your goal is to help the user with their questions and tasks, introduce yourself as Andelina when they first start the conversation."
    if self.provider == "anthropic":
      messages = []
      if self.conversation_history:
        messages.extend(self.conversation_history[-3:])
      
      messages.append(
        {
          "role": "user",
          "content": user_message
        }
      )
      
      response = self.client.messages.create(
        model=self.model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=messages
      )
      
      assistant_message = response.content[0].text
      
      self.conversation_history.append(
        {
          "role": "user",
          "content": user_message
        }
      )
      self.conversation_history.append(
        {
          "role": "assistant",
          "content": assistant_message
        }
      )
    else:
      messages = []
      if system_prompt:
        messages.append(
          {
            "role": "system", 
            "content": system_prompt
          }
        )
      if self.conversation_history:
        messages.extend(self.conversation_history[-3:])
        
      messages.append(
        {
          "role": "user",
          "content": user_message
        }
      )
      
      if self.provider == "openai":
        api_params = {
          "model": self.model,
          "messages": messages,
          "max_completion_tokens": max_tokens
        }
        if not self.model.startswith("gpt-5"):
          api_params["temperature"] = temperature
        
        response = self.client.chat.completions.create(**api_params)
      else:
        response = self.client.chat.completions.create(
          model=self.model,
          messages=messages,
          temperature=temperature,
          max_tokens=max_tokens
        )
      
      assistant_message = response.choices[0].message.content
      
      self.conversation_history.append(
        {
          "role": "user",
          "content": user_message
        }
      )
      self.conversation_history.append(
        {
          "role": "assistant",
          "content": assistant_message
        }
      )
    
    return assistant_message
  
  # def clear_history(self):
  # def get_history(self):
    
if __name__ == "__main__":
  llm_app = LLMApp(
    api_key=app_config.groq_api_key, 
    model_name="llama-3.3-70b-versatile",
    provider="groq"
  )
  while True:
    message = input("Enter your message: ")
    response = llm_app.chat(message)
    print(f"Assistant Response: {response}")
      