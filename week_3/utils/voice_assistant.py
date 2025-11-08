from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

from .knowledge_base import KnowledgeBase


class VoiceAssistant:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        llm_type: str = "openai",
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        """Initialize voice assistant with RAG"""
        self.knowledge_base = knowledge_base
        
        # Initialize LLM
        if llm_type == "openai":
            if not api_key:
                raise ValueError("OpenAI API key required")
            self.llm = ChatOpenAI(
                model_name=llm_model,
                temperature=temperature,
                openai_api_key=api_key
            )
        elif llm_type == "ollama":
            self.llm = ChatOllama(
                model=llm_model,
                temperature=temperature,
                timeout=120
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
        
        # System prompt
        self.system_prompt = system_prompt or """You are a helpful voice assistant. 
Answer questions based on the provided context from the knowledge base.
Be conversational and natural in your responses since this is a voice interaction.
Keep responses concise but informative.
If the answer is not in the context, say so politely."""
        
        # Conversation history
        self.conversation_history: List = []
    
    def _format_context(self, docs) -> str:
        """Format retrieved documents as context"""
        if not docs:
            return "No relevant information found in knowledge base."
        return "\n\n".join([doc.page_content for doc in docs])
    
    def _format_history(self) -> str:
        """Format conversation history"""
        if not self.conversation_history:
            return "No previous conversation."
        
        history_text = []
        for msg in self.conversation_history[-6:]:  # Last 3 exchanges
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_text.append(f"{role}: {msg.content}")
        return "\n".join(history_text)
    
    def query(self, question: str) -> str:
        """
        Query the assistant with RAG
        Returns response text
        """
        # Retrieve relevant documents
        docs = self.knowledge_base.query(question, k=4)
        
        # Create prompt
        template = """{system_prompt}

Context from knowledge base:
{context}

Conversation history:
{history}

User question: {question}

Assistant response:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create chain
        chain = (
            {
                "system_prompt": lambda x: self.system_prompt,
                "context": lambda x: self._format_context(docs),
                "history": lambda x: self._format_history(),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Get response
        response = chain.invoke(question)
        
        # Update history
        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content=response))
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[dict]:
        """Get conversation history as list of dicts"""
        history = []
        for msg in self.conversation_history:
            history.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            })
        return history

