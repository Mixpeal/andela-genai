from langgraph.graph import StateGraph, START, END
from .blog_state import BlogState
from .blog_nodes import BlogNodes


class BlogGraphBuilder:
    """Build the blog generation workflow graph"""
    
    def __init__(self, llm):
        self.llm = llm
        self.graph = StateGraph(BlogState)
        self.nodes = BlogNodes(llm)
    
    def build_graph(self):
        """Build the complete blog generation workflow"""
        
        # Add nodes
        self.graph.add_node("generate_title", self.nodes.generate_title)
        self.graph.add_node("generate_outline", self.nodes.generate_outline)
        self.graph.add_node("generate_keywords", self.nodes.generate_keywords)
        self.graph.add_node("generate_content", self.nodes.generate_content)
        
        # Add edges to create the workflow
        self.graph.add_edge(START, "generate_title")
        self.graph.add_edge("generate_title", "generate_outline")
        self.graph.add_edge("generate_outline", "generate_keywords")
        self.graph.add_edge("generate_keywords", "generate_content")
        self.graph.add_edge("generate_content", END)
        
        return self.graph.compile()

