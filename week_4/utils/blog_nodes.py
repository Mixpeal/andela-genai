from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from .blog_state import BlogState, Blog


class BlogNodes:
    """Nodes for blog generation workflow"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_title(self, state: BlogState) -> dict:
        """Generate a creative and SEO-friendly blog title"""
        topic = state.get("topic", "")
        extracted_content = state.get("extracted_content")
        
        if not topic:
            return {"error": "No topic provided"}
        
        # Build context from extracted content if available
        context = ""
        if extracted_content:
            plain_text = extracted_content.get("plain_text", "")
            sources = extracted_content.get("sources", [])
            
            if plain_text:
                context += f"\n\nAdditional Context: {plain_text}"
            
            if sources:
                context += "\n\nReference Sources:"
                for idx, source in enumerate(sources, 1):
                    source_type = source.get("content_type")
                    source_title = source.get("title", "Unknown")
                    source_content = source.get("content", "")[:300]
                    
                    if source_type == "youtube":
                        context += f"\n\n{idx}. YouTube Video: {source_title}\nExcerpt: {source_content}..."
                    elif source_type == "article":
                        context += f"\n\n{idx}. Article: {source_title}\nExcerpt: {source_content}..."
        
        prompt = ChatPromptTemplate.from_template(
            """You are an expert blog content writer. Generate a creative and SEO-friendly blog title for the following topic:

Topic: {topic}{context}

Requirements:
- Make it engaging and click-worthy
- Keep it concise (60-70 characters)
- Include relevant keywords
- Make it clear what the blog is about
- If multiple sources are provided, synthesize them into a cohesive title

Return ONLY the title, nothing else."""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        # Stream the title generation
        title_chunks = []
        for chunk in chain.stream({"topic": topic, "context": context}):
            title_chunks.append(chunk)
        
        title = "".join(title_chunks)
        return {"blog": Blog(title=title.strip())}
    
    def generate_outline(self, state: BlogState) -> dict:
        """Generate a structured outline for the blog"""
        topic = state.get("topic", "")
        blog = state.get("blog")
        extracted_content = state.get("extracted_content")
        
        if not blog or not blog.title:
            return {"error": "No title found"}
        
        # Build comprehensive reference material from all sources
        reference = ""
        if extracted_content:
            plain_text = extracted_content.get("plain_text", "")
            sources = extracted_content.get("sources", [])
            
            if plain_text:
                reference += f"\n\nAdditional Context: {plain_text}"
            
            if sources:
                reference += "\n\nReference Materials:"
                for idx, source in enumerate(sources, 1):
                    source_type = source.get("content_type")
                    source_title = source.get("title", "Unknown")
                    source_content = source.get("content", "")[:2000]
                    source_url = source.get("url", "")
                    
                    if source_type == "youtube":
                        reference += f"\n\n{idx}. YouTube Video: '{source_title}'\nURL: {source_url}\nTranscript:\n{source_content}..."
                    elif source_type == "article":
                        reference += f"\n\n{idx}. Article: '{source_title}'\nURL: {source_url}\nContent:\n{source_content}..."
        
        prompt = ChatPromptTemplate.from_template(
            """You are an expert blog content writer. Create a detailed outline for a blog post.

Topic: {topic}
Title: {title}{reference}

Create a structured outline with:
- Introduction
- 3-5 main sections with subsections
- Conclusion

Use markdown formatting with headers (##, ###).
Be specific about what each section will cover.
If multiple reference materials are provided, synthesize insights from all of them into the outline.

OUTLINE:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        outline = chain.invoke({"topic": topic, "title": blog.title, "reference": reference})
        
        # Update blog with outline
        updated_blog = Blog(
            title=blog.title,
            outline=outline.strip()
        )
        
        return {"blog": updated_blog}
    
    def generate_keywords(self, state: BlogState) -> dict:
        """Generate SEO keywords for the blog"""
        topic = state.get("topic", "")
        blog = state.get("blog")
        extracted_content = state.get("extracted_content")
        
        if not blog:
            return {"error": "No blog found"}
        
        # Add context from all extracted sources
        context = ""
        if extracted_content:
            plain_text = extracted_content.get("plain_text", "")
            sources = extracted_content.get("sources", [])
            
            if plain_text:
                context += f"\nAdditional context: {plain_text}"
            
            if sources:
                titles = [s.get('title', '') for s in sources]
                context += f"\nReference sources: {', '.join(titles)}"
        
        prompt = ChatPromptTemplate.from_template(
            """You are an SEO expert. Generate relevant SEO keywords for this blog post.

Topic: {topic}
Title: {title}{context}

Generate 10-15 relevant keywords and phrases that should be targeted in this blog post.
Include both short-tail and long-tail keywords.
If multiple sources are referenced, include keywords relevant to all of them.
Return them as a comma-separated list.

KEYWORDS:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        keywords = chain.invoke({"topic": topic, "title": blog.title, "context": context})
        
        # Update blog with keywords
        updated_blog = Blog(
            title=blog.title,
            outline=blog.outline,
            seo_keywords=keywords.strip()
        )
        
        return {"blog": updated_blog}
    
    def generate_content(self, state: BlogState) -> dict:
        """Generate the full blog content"""
        topic = state.get("topic", "")
        blog = state.get("blog")
        extracted_content = state.get("extracted_content")
        
        if not blog or not blog.outline:
            return {"error": "No outline found"}
        
        # Build comprehensive reference material from all sources
        reference = ""
        source_attribution = ""
        
        if extracted_content:
            plain_text = extracted_content.get("plain_text", "")
            sources = extracted_content.get("sources", [])
            
            if plain_text:
                reference += f"\n\nAdditional Context: {plain_text}"
            
            if sources:
                reference += "\n\nReference Materials:"
                source_titles = []
                
                for idx, source in enumerate(sources, 1):
                    source_type = source.get("content_type")
                    source_title = source.get("title", "Unknown")
                    source_content = source.get("content", "")
                    source_url = source.get("url", "")
                    
                    source_titles.append(source_title)
                    
                    if source_type == "youtube":
                        reference += f"\n\n{idx}. YouTube Video:\nTitle: {source_title}\nURL: {source_url}\n\nTranscript:\n{source_content}"
                    
                    elif source_type == "article":
                        authors = source.get("authors", [])
                        author_text = f" by {', '.join(authors)}" if authors else ""
                        reference += f"\n\n{idx}. Article:\nTitle: {source_title}{author_text}\nURL: {source_url}\n\nContent:\n{source_content}"
                
                if len(sources) == 1:
                    source_attribution = f"\n\nNote: Include a reference to '{source_titles[0]}' in the blog post where relevant."
                else:
                    source_attribution = f"\n\nNote: Synthesize insights from all {len(sources)} sources and cite them naturally throughout the blog post."
        
        prompt = ChatPromptTemplate.from_template(
            """You are an expert blog content writer. Write a comprehensive and engaging blog post.

Topic: {topic}
Title: {title}
Outline: {outline}
SEO Keywords to include: {keywords}{reference}{source_attribution}

Write a full blog post following the outline. 

Requirements:
- Use markdown formatting
- Write in a professional yet conversational tone
- Include the SEO keywords naturally throughout the content
- Make each section detailed and informative (aim for 1500-2000 words total)
- If reference materials are provided, incorporate insights, data, and information from ALL of them
- Synthesize multiple sources into a cohesive narrative
- Include examples, tips, or insights where relevant
- Add a compelling introduction and conclusion
- Use bullet points and numbered lists where appropriate
- Cite sources naturally in the content

BLOG CONTENT:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        content = chain.invoke({
            "topic": topic,
            "title": blog.title,
            "outline": blog.outline,
            "keywords": blog.seo_keywords,
            "reference": reference,
            "source_attribution": source_attribution
        })
        
        # Update blog with content
        updated_blog = Blog(
            title=blog.title,
            outline=blog.outline,
            seo_keywords=blog.seo_keywords,
            content=content.strip()
        )
        
        return {"blog": updated_blog}

