# AI Blog Generator

A powerful blog generation application using LangGraph and OpenAI that creates high-quality, SEO-optimized blog posts from any topic.

## Features

- 🎯 **Topic-Based Generation**: Enter any topic and get a complete blog post
- 📹 **YouTube Integration**: Paste YouTube URL(s) to generate a blog from video content (with automatic transcription)
- 📄 **Article Scraping**: Paste article URL(s) to generate a blog based on that content
- 🔗 **Multiple Sources**: Use multiple URLs in one message! Mix YouTube videos and articles
- 📝 **Multi-Step Workflow**: Uses LangGraph to orchestrate content extraction, title, outline, keywords, and content generation
- 🔍 **SEO Optimized**: Automatically generates relevant SEO keywords
- ✍️ **High-Quality Content**: Produces professional, engaging blog posts (1500-2000 words)
- 📋 **Structured Outline**: View the blog structure before diving into content
- 📥 **Export to Markdown**: Download your blog posts in markdown format
- 🎨 **Beautiful UI**: Clean Streamlit interface with real-time progress updates
- 🤖 **Smart Content Integration**: Automatically incorporates insights from source materials

## Architecture

The application uses a clean, modular architecture inspired by best practices:

```
week_4/
├── utils/                      # Core utilities
│   ├── __init__.py
│   ├── llm_provider.py        # OpenAI LLM provider
│   ├── blog_state.py          # Pydantic models and state definition
│   ├── blog_nodes.py          # LangGraph nodes for each generation step
│   ├── blog_graph.py          # LangGraph workflow builder
│   └── blog_generator.py      # Main generator class
├── streamlit_app.py           # Streamlit UI
├── appconfig.py               # Configuration management
├── requirements.txt           # Python dependencies
└── README.md
```

## Workflow

The blog generation follows an intelligent LangGraph workflow:

### For Text Topics:
1. **Title Generation**: Creates a creative, SEO-friendly title
2. **Outline Creation**: Structures the blog with sections and subsections
3. **Keyword Generation**: Identifies 10-15 relevant SEO keywords
4. **Content Writing**: Produces the full blog post with all sections

### For YouTube/Article URLs:
1. **Content Extraction**: Downloads and transcribes YouTube videos OR scrapes article content
2. **Title Generation**: Creates a title based on the extracted content
3. **Outline Creation**: Structures the blog using insights from the source
4. **Keyword Generation**: Identifies keywords relevant to both topic and source
5. **Content Writing**: Produces a comprehensive blog incorporating source material

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_TEMPERATURE=0.7
```

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

Then:
1. Enter your OpenAI API key in the sidebar (or set it in `.env`)
2. The generator initializes automatically!
3. Enter a topic, YouTube URL, or article URL in the chat input
4. Watch as the AI extracts content (if URL) and generates your blog post step by step
5. Download the blog as markdown when complete

### Example Inputs:
- **Text topic**: "The Future of AI in Healthcare"
- **Single YouTube URL**: `https://www.youtube.com/watch?v=example`
- **Single Article URL**: `https://example.com/article`
- **Topic + URL**: "Write about sustainable energy https://example.com/article"
- **Multiple URLs**: `https://youtube.com/watch?v=1 https://example.com/article1 https://example.com/article2`
- **Topic + Multiple URLs**: "Compare AI approaches https://youtube.com/watch?v=1 https://example.com/article"

## Models Supported

- GPT-4o-mini (default, recommended for speed and cost)
- GPT-4o
- GPT-4
- GPT-3.5-turbo

## Configuration

Adjust settings in the sidebar:
- **Model**: Choose your preferred OpenAI model
- **Temperature**: Control creativity (0 = focused, 1 = creative)

## Technologies Used

- **LangChain**: Framework for LLM applications
- **LangGraph**: Workflow orchestration for multi-step generation
- **OpenAI**: GPT models for content generation
- **Streamlit**: Web interface
- **Pydantic**: Data validation and modeling

## License

MIT License

