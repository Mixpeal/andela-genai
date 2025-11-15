from typing import TypedDict, Optional, Dict
from pydantic import BaseModel, Field


class Blog(BaseModel):
    """Blog content model"""
    title: str = Field(description="The title of the blog post")
    outline: str = Field(default="", description="The outline/structure of the blog post")
    content: str = Field(default="", description="The main content of the blog post")
    seo_keywords: str = Field(default="", description="SEO keywords for the blog post")


class BlogState(TypedDict):
    """State for blog generation workflow"""
    topic: str
    blog: Optional[Blog]
    error: Optional[str]
    extracted_content: Optional[Dict]  # Content from URLs (YouTube/articles) - can contain multiple sources

