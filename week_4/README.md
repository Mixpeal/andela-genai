# AI Blog Generator

A blog generation application using LangGraph and OpenAI that creates high-quality, SEO-optimized blog posts from any topic.

## Features

- 🎯 **Topic-Based Generation**: Enter any topic and get a complete blog post
- 📹 **YouTube Integration**: Paste YouTube URL(s) to generate a blog from video content (with automatic transcription)
- 📄 **Article Scraping**: Paste article URL(s) to generate a blog based on that content


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

## License

MIT License

