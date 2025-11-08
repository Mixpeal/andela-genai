import streamlit as st
import os
import threading
from pathlib import Path
from utils.knowledge_base import KnowledgeBase
from utils.voice_assistant import VoiceAssistant
from appconfig import app_config

# Try to import audio handler (optional)
try:
    from utils.audio_handler import AudioHandler
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    AUDIO_ERROR = str(e)

st.set_page_config(
    page_title="Voice Assistant", page_icon="🎙️", layout="wide"
)

# Initialize session state
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None

if "voice_assistant" not in st.session_state:
    st.session_state.voice_assistant = None

if "audio_handler" not in st.session_state:
    st.session_state.audio_handler = None

if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []

if "is_session_active" not in st.session_state:
    st.session_state.is_session_active = False

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "voice_thread" not in st.session_state:
    st.session_state.voice_thread = None

if "stop_voice_loop" not in st.session_state:
    st.session_state.stop_voice_loop = {"stop": False}

if "voice_status" not in st.session_state:
    st.session_state.voice_status = {"status": "idle", "message": ""}


def initialize_knowledge_base(embedding_type, api_key):
    """Initialize knowledge base"""
    try:
        kb = KnowledgeBase(
            embedding_type=embedding_type,
            api_key=api_key if embedding_type == "openai" else None
        )
        return kb, None
    except Exception as e:
        return None, str(e)


def initialize_voice_assistant(kb, llm_type, llm_model, api_key, temperature):
    """Initialize voice assistant"""
    try:
        va = VoiceAssistant(
            knowledge_base=kb,
            llm_type=llm_type,
            llm_model=llm_model,
            api_key=api_key if llm_type == "openai" else None,
            temperature=temperature
        )
        return va, None
    except Exception as e:
        return None, str(e)


def initialize_audio_handler(elevenlabs_key, openai_key, voice_id, language="en", tts_provider="openai"):
    """Initialize audio handler"""
    if not AUDIO_AVAILABLE:
        return None, "Audio not available. Install PortAudio: brew install portaudio"
    
    try:
        ah = AudioHandler(
            elevenlabs_api_key=elevenlabs_key,
            openai_api_key=openai_key,
            voice_id=voice_id,
            language=language,
            tts_provider=tts_provider
        )
        return ah, None
    except Exception as e:
        return None, str(e)


def voice_interaction_loop(audio_handler, voice_assistant, conversation_log, stop_flag, status_dict):
    """Main voice interaction loop - runs in background thread"""
    try:
        audio_handler.start_listening()
        status_dict['status'] = 'listening'
        status_dict['message'] = '🎤 Listening...'
        
        while not stop_flag['stop']:
            try:
                # Listen for speech with VAD
                status_dict['status'] = 'listening'
                status_dict['message'] = '🎤 Listening for your voice...'
                
                audio_data = audio_handler.listen_with_vad(silence_duration=1.5)
                
                if audio_data and not stop_flag['stop']:
                    # Transcribe
                    status_dict['status'] = 'transcribing'
                    status_dict['message'] = '📝 Transcribing...'
                    
                    question = audio_handler.transcribe_audio(audio_data)
                    
                    if question.strip():
                        # Add to conversation log
                        conversation_log.append({
                            "role": "user",
                            "content": question
                        })
                        
                        # Get response from RAG
                        status_dict['status'] = 'processing'
                        status_dict['message'] = f'🤔 Processing: "{question}"'
                        
                        response = voice_assistant.query(question)
                        
                        # Add to conversation log
                        conversation_log.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                        # Speak response
                        status_dict['status'] = 'speaking'
                        status_dict['message'] = '🔊 Speaking response...'
                        
                        audio_bytes = audio_handler.text_to_speech(response)
                        audio_handler.play_audio(audio_bytes)
            except Exception as e:
                if stop_flag['stop']:
                    break
                print(f"Loop iteration error: {e}")
                continue
                    
    except Exception as e:
        print(f"Voice loop error: {e}")
        status_dict['status'] = 'error'
        status_dict['message'] = f'❌ Error: {str(e)}'
    finally:
        try:
            audio_handler.stop_listening()
        except:
            pass
        stop_flag['stop'] = True
        status_dict['status'] = 'idle'
        status_dict['message'] = ''


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Keys
    st.subheader("API Keys")
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=app_config.openai_api_key or ""
    )
    
    elevenlabs_key = st.text_input(
        "ElevenLabs API Key",
        type="password",
        value=app_config.elevenlabs_api_key or ""
    )
    
    st.divider()
    
    # Model Settings
    st.subheader("Model Settings")
    llm_type = st.selectbox(
        "LLM Provider",
        ["openai", "ollama"],
        index=0
    )
    
    if llm_type == "openai":
        llm_model = st.selectbox(
            "LLM Model",
            ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
            index=0
        )
    else:
        llm_model = st.text_input("Ollama Model", value="llama3.2")
    
    embedding_type = st.selectbox(
        "Embedding Provider",
        ["openai", "chroma"],
        index=0
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1
    )
    
    st.divider()
    
    # Voice Settings
    st.subheader("Voice Settings")
    
    # TTS Provider selection
    tts_provider = st.selectbox(
        "TTS Provider",
        ["openai", "elevenlabs"],
        index=0,
        help="Text-to-Speech provider"
    )
    
    # Language selection
    language = st.selectbox(
        "Language",
        ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"],
        index=0,
        help="Language for speech recognition"
    )
    
    # Get available voices if audio handler exists
    available_voices = []
    if st.session_state.audio_handler:
        available_voices = st.session_state.audio_handler.get_available_voices()
    
    if available_voices:
        voice_options = {name: vid for vid, name in available_voices}
        selected_voice_name = st.selectbox(
            "Voice",
            options=list(voice_options.keys()),
            index=0
        )
        voice_id = voice_options[selected_voice_name]
    else:
        voice_id = st.text_input(
            "Voice ID",
            value="21m00Tcm4TlvDq8ikWAM",
            help="Default: Rachel"
        )
    
    st.divider()
    
    # Initialize button
    if st.button("🔄 Initialize System", use_container_width=True):
        with st.spinner("Initializing..."):
            # Initialize KB
            kb, error = initialize_knowledge_base(embedding_type, openai_key)
            if error:
                st.error(f"KB Error: {error}")
            else:
                st.session_state.knowledge_base = kb
                
                # Initialize VA
                va, error = initialize_voice_assistant(
                    kb, llm_type, llm_model, openai_key, temperature
                )
                if error:
                    st.error(f"VA Error: {error}")
                else:
                    st.session_state.voice_assistant = va
                    
                    # Initialize Audio
                    ah, error = initialize_audio_handler(
                        elevenlabs_key, openai_key, voice_id, language, tts_provider
                    )
                    if error:
                        st.error(f"Audio Error: {error}")
                    else:
                        st.session_state.audio_handler = ah
                        st.success("✅ System Ready!")
    
    st.divider()
    
    # Knowledge Base Info
    if st.session_state.knowledge_base:
        sources = st.session_state.knowledge_base.get_all_sources()
        st.subheader(f"📚 Knowledge Base ({len(sources)})")
        if sources:
            for source in sources:
                st.caption(f"• {source}")
        
        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            st.session_state.knowledge_base.clear()
            st.session_state.uploaded_files = []
            st.success("Knowledge base cleared!")
            st.rerun()


# Main Content
st.title("🎙️ Voice Assistant")

# Show audio status
if not AUDIO_AVAILABLE:
    st.warning(f"⚠️ Audio features unavailable. Install PortAudio: `brew install portaudio && pip install --upgrade --force-reinstall pyaudio`")

# Create tabs
tab1, tab2 = st.tabs(["📚 Knowledge Base", "🎤 Voice Chat"])

# Knowledge Base Tab
with tab1:
    st.header("Knowledge Base Management")
    st.markdown("Upload documents to train your voice assistant")
    
    if not st.session_state.knowledge_base:
        st.warning("⚠️ Initialize system in sidebar first")
    else:
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.button("📥 Process Files", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                upload_dir = "data/voice_kb/uploads"
                os.makedirs(upload_dir, exist_ok=True)
                
                total_chunks = 0
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save file
                    file_path = os.path.join(upload_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Get file type
                    file_ext = Path(uploaded_file.name).suffix[1:].lower()
                    
                    # Process file
                    try:
                        chunks = st.session_state.knowledge_base.add_file(
                            file_path, file_ext
                        )
                        total_chunks += chunks
                        st.session_state.uploaded_files.append({
                            "name": uploaded_file.name,
                            "chunks": chunks
                        })
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                st.success(f"✅ Processed {len(uploaded_files)} files ({total_chunks} chunks)")
                st.rerun()
        
        # Display uploaded files
        if st.session_state.uploaded_files:
            st.divider()
            st.subheader("Uploaded Files")
            for file_info in st.session_state.uploaded_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {file_info['name']}")
                with col2:
                    st.caption(f"{file_info['chunks']} chunks")

# Voice Chat Tab
with tab2:
    st.header("Voice Chat Session")
    
    if not st.session_state.voice_assistant or not st.session_state.audio_handler:
        st.warning("⚠️ Initialize system and upload documents first")
    else:
        # Session controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not st.session_state.is_session_active:
                if st.button("🎙️ Start Session", type="primary", use_container_width=True):
                    if not AUDIO_AVAILABLE:
                        st.error("Audio not available!")
                    else:
                        st.session_state.is_session_active = True
                        st.session_state.stop_voice_loop = {"stop": False}
                        
                        # Start voice loop in background thread
                        voice_thread = threading.Thread(
                            target=voice_interaction_loop,
                            args=(
                                st.session_state.audio_handler,
                                st.session_state.voice_assistant,
                                st.session_state.conversation_log,
                                st.session_state.stop_voice_loop,
                                st.session_state.voice_status
                            ),
                            daemon=True
                        )
                        voice_thread.start()
                        st.session_state.voice_thread = voice_thread
                        st.rerun()
            else:
                if st.button("⏹️ Stop Session", type="secondary", use_container_width=True):
                    try:
                        st.session_state.is_session_active = False
                        st.session_state.stop_voice_loop["stop"] = True
                        if st.session_state.audio_handler:
                            st.session_state.audio_handler.stop_listening()
                            st.session_state.audio_handler.stop_speaking()
                    except Exception as e:
                        print(f"Error stopping session: {e}")
                    finally:
                        st.rerun()
        
        with col2:
            if st.button("🔇 Interrupt", use_container_width=True):
                if st.session_state.audio_handler:
                    st.session_state.audio_handler.stop_speaking()
        
        with col3:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.conversation_log = []
                if st.session_state.voice_assistant:
                    st.session_state.voice_assistant.clear_history()
                st.rerun()
        
        st.divider()
        
        # Status indicator
        status_placeholder = st.empty()
        
        if st.session_state.is_session_active:
            status = st.session_state.voice_status.get('status', 'idle')
            message = st.session_state.voice_status.get('message', '')
            
            if status == 'listening':
                status_placeholder.success(message)
            elif status == 'transcribing':
                status_placeholder.info(message)
            elif status == 'processing':
                status_placeholder.warning(message)
            elif status == 'speaking':
                status_placeholder.info(message)
            elif status == 'error':
                status_placeholder.error(message)
            else:
                status_placeholder.info("🎙️ Session Active")
        else:
            status_placeholder.caption("Session inactive")
        
        st.divider()
        
        # Conversation log
        st.subheader("Conversation Log")
        
        if st.session_state.conversation_log:
            for entry in st.session_state.conversation_log:
                with st.chat_message(entry["role"]):
                    st.markdown(entry["content"])
        else:
            st.caption("No conversation yet. Start a session and speak!")
        
        # Auto-refresh during active session
        if st.session_state.is_session_active:
            import time
            time.sleep(1)
            st.rerun()

# Footer
st.divider()
st.caption("🎙️ Voice Assistant powered by ElevenLabs, OpenAI Whisper, and LangChain")

