# Week 3 - AI Content Summarizer & Voice Assistant

Two powerful AI applications: Omni Chat for content summarization and Voice Assistant for document-based conversations.

## Setup

### 1. Install Dependencies

```bash
# Install system dependencies (macOS)
brew install portaudio ffmpeg

# Install Python packages
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file in the `week_3` directory:

```bash
OPENAI_API_KEY=your_openai_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here  # Optional, for voice features
```

### 3. Run Applications

**Omni Chat** (Port 8501):
```bash
streamlit run streamlit_app.py
```

**Voice Assistant** (Port 8502):
```bash
streamlit run voice_assistant_app.py --server.port 8502
```

## Applications

### 🤖 Omni Chat (`streamlit_app.py`)

Intelligent content processor that automatically detects and handles:
- **YouTube Videos**: Downloads, transcribes, summarizes, enables Q&A
- **Articles**: Extracts, summarizes, enables Q&A
- **Text Questions**: Uses RAG over processed content

**Features:**
- Auto-detection of content type
- Real-time progress tracking
- Persistent context across conversation
- Multi-source Q&A

### 🎙️ Voice Assistant (`voice_assistant_app.py`)

Voice-based RAG system with document knowledge base.

**Features:**
- Upload PDF, TXT, MD, DOCX files
- Voice Activity Detection (VAD)
- Real-time voice interaction
- ElevenLabs text-to-speech
- OpenAI Whisper transcription

**Note:** Voice features require working PyAudio. If audio fails, knowledge base features still work.

## Troubleshooting

### PyAudio Issues (macOS)

```bash
# Reinstall with proper flags
pip uninstall -y pyaudio
LDFLAGS="-L/opt/homebrew/lib" CFLAGS="-I/opt/homebrew/include" pip install pyaudio
```

### FFmpeg Missing

```bash
brew install ffmpeg
```


