from typing import Generator, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage
from .llm_provider import LLMProvider
from .blog_graph import BlogGraphBuilder
from .blog_state import Blog
from .content_extractor import ContentExtractor
from .chat_classifier import ChatClassifier


class BlogGenerator:
    """Main blog generation class using LangGraph"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        whisper_model: str = "base"
    ):
        """Initialize blog generator with OpenAI LLM"""
        self.llm_provider = LLMProvider(api_key, model_name, temperature)
        self.llm = self.llm_provider.get_llm()
        self.graph_builder = BlogGraphBuilder(self.llm)
        self.graph = self.graph_builder.build_graph()
        self.content_extractor = ContentExtractor(whisper_model)
        self.chat_classifier = ChatClassifier(self.llm)
        self.generated_blogs = []
        self.conversation_history = []
    
    def generate_blog(self, topic: str) -> Generator[Dict, None, None]:
        """
        Generate a blog post from a topic with progress updates
        Yields status updates for real-time display in Streamlit
        """
        try:
            yield {"status": "started", "message": f"Starting blog generation..."}
            
            # Check if topic contains URLs and extract all content
            urls = self.content_extractor.detect_all_urls(topic)
            extracted_content = None
            
            if urls:
                total_urls = len(urls)
                url_types = [u['type'] for u in urls]
                youtube_count = url_types.count('youtube')
                article_count = url_types.count('article')
                
                summary = []
                if youtube_count > 0:
                    summary.append(f"{youtube_count} YouTube video{'s' if youtube_count > 1 else ''}")
                if article_count > 0:
                    summary.append(f"{article_count} article{'s' if article_count > 1 else ''}")
                
                yield {"status": "extracting", "message": f"📥 Found {total_urls} URL(s): {' and '.join(summary)}"}
                
                for update in self.content_extractor.extract_all_content(topic):
                    status = update.get("status")
                    message = update.get("message", "")
                    
                    if status == "error":
                        yield {"status": "error", "message": message}
                        return
                    
                    elif status == "complete":
                        extracted_content = update
                        total_sources = update.get("total_sources", 0)
                        if total_sources > 0:
                            yield {"status": "extracted", "message": f"✅ Successfully extracted content from {total_sources} source(s)!"}
                    
                    else:
                        # Pass through progress updates
                        yield {"status": status, "message": message}
            else:
                # No URLs, just plain text topic
                plain_text = self.content_extractor.get_plain_text(topic)
                extracted_content = {
                    "content_type": "text",
                    "plain_text": plain_text,
                    "sources": []
                }
            
            yield {"status": "title", "message": "📝 Generating creative title..."}
            
            # Run the graph step by step to provide updates
            state = {
                "topic": topic,
                "blog": None,
                "error": None,
                "extracted_content": extracted_content
            }
            
            # We'll invoke the full graph but yield updates
            yield {"status": "outline", "message": "📋 Creating blog outline..."}
            
            yield {"status": "keywords", "message": "🔍 Generating SEO keywords..."}
            
            yield {"status": "content", "message": "✍️ Writing blog content (this may take a moment)..."}
            
            # Execute the full graph
            result = self.graph.invoke(state)
            
            if result.get("error"):
                yield {"status": "error", "message": result["error"]}
                return
            
            blog = result.get("blog")
            
            if not blog:
                yield {"status": "error", "message": "Failed to generate blog"}
                return
            
            # Store the generated blog
            self.generated_blogs.append({
                "topic": topic,
                "blog": blog,
                "extracted_content": extracted_content
            })
            
            yield {
                "status": "complete",
                "message": "Blog generated successfully!",
                "blog": blog,
                "extracted_content": extracted_content
            }
            
        except Exception as e:
            yield {"status": "error", "message": f"Error generating blog: {str(e)}"}
    
    def get_generated_blogs(self):
        """Get all generated blogs"""
        return self.generated_blogs
    
    def clear_history(self):
        """Clear generated blogs history"""
        self.generated_blogs = []
        self.conversation_history = []
    
    def chat(self, message: str) -> Generator[str, None, None]:
        """
        Chat with the AI (streaming response)
        Yields chunks of text as they're generated
        """
        from langchain_core.messages import SystemMessage
        
        # Build messages with system context
        messages = []
        
        # Add system message with context about generated blogs
        system_context = """You are a helpful AI assistant for a blog generation application. 
You can help users understand how the app works, answer questions about their generated blogs, 
and have casual conversations. Be friendly, concise, and helpful."""
        
        # Add info about generated blogs if any exist
        if self.generated_blogs:
            system_context += f"\n\nThe user has generated {len(self.generated_blogs)} blog(s) in this session:"
            for idx, blog_data in enumerate(self.generated_blogs[-3:], 1):  # Last 3 blogs
                blog = blog_data.get("blog")
                if blog:
                    system_context += f"\n{idx}. Title: '{blog.title}'"
        
        messages.append(SystemMessage(content=system_context))
        
        # Add recent conversation history
        for msg in self.conversation_history[-10:]:  # Last 5 exchanges
            messages.append(msg)
        
        # Add current message
        messages.append(HumanMessage(content=message))
        
        # Stream response
        full_response = ""
        for chunk in self.llm.stream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                full_response += chunk.content
                yield chunk.content
        
        # Update conversation history
        self.conversation_history.append(HumanMessage(content=message))
        self.conversation_history.append(AIMessage(content=full_response))
    
    def export_blog_markdown(self, blog: Blog) -> str:
        """Export blog as markdown format"""
        markdown = f"# {blog.title}\n\n"
        
        if blog.seo_keywords:
            markdown += f"**Keywords:** {blog.seo_keywords}\n\n"
        
        markdown += "---\n\n"
        markdown += blog.content
        
        return markdown

