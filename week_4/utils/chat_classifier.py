from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal


class UserIntent(BaseModel):
    """Classification of user intent"""
    intent: Literal["generate_blog", "chat", "question_about_blog"] = Field(
        description="The user's intent: 'generate_blog' for blog generation requests, 'chat' for general conversation, 'question_about_blog' for questions about previously generated content"
    )
    reasoning: str = Field(description="Brief reasoning for the classification")


class ChatClassifier:
    """Classify user messages to determine intent"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def classify_message(self, message: str, has_generated_blogs: bool = False) -> UserIntent:
        """
        Classify user message intent
        
        Args:
            message: The user's message
            has_generated_blogs: Whether there are previously generated blogs in the session
        
        Returns:
            UserIntent with classification and reasoning
        """
        
        context = ""
        if has_generated_blogs:
            context = "\nNote: The user has previously generated blog(s) in this session."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that classifies user intent in a blog generation application.

Classify the user's message into one of these categories:

1. **generate_blog**: User wants to generate a new blog post
   - Contains a topic they want a blog about
   - Contains URL(s) to use as source material
   - Explicit requests like "write a blog", "create a post", "generate content"
   - Even simple topics like "AI in healthcare" should be classified as generate_blog

2. **chat**: General conversation or greetings
   - Greetings: "hi", "hello", "hey"
   - General questions: "what can you do?", "how does this work?"
   - Casual conversation
   
3. **question_about_blog**: Questions about previously generated blog content
   - Only if blogs have been generated in this session
   - Questions like "can you explain that section?", "what about the keywords?", "change the title"
   - References to previous content

Default to 'generate_blog' if the message contains a clear topic or URL, even if phrased casually.{context}

Respond with your classification and brief reasoning."""),
            ("user", "{message}")
        ])
        
        # Use structured output
        structured_llm = self.llm.with_structured_output(UserIntent)
        chain = prompt | structured_llm
        
        result = chain.invoke({
            "message": message,
            "context": context
        })
        
        return result

