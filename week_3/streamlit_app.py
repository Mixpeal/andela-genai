import streamlit as st
import uuid
from utils import OmniSummarizer
from appconfig import app_config

st.set_page_config(
    page_title="Omni Summarizer", page_icon="🤖", layout="centered"
)

# Initialize session state
if "omni_summarizer" not in st.session_state:
    st.session_state.omni_summarizer = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def initialize_omni_summarizer(
    llm_type, llm_model, embedding_type, api_key, temperature, whisper_model="base"
):
    """Initialize omni summarizer"""
    try:
        omni_summarizer = OmniSummarizer(
            llm_type=llm_type,
            llm_model_name=llm_model,
            embedding_type=embedding_type,
            api_key=api_key if llm_type == "openai" or embedding_type == "openai" else None,
            temperature=temperature,
            whisper_model=whisper_model,
        )
        return omni_summarizer, None
    except Exception as e:
        return None, str(e)


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=app_config.openai_api_key or "",
    )
    if not api_key:
        api_key = app_config.openai_api_key

    llm_type = st.selectbox(
        "LLM Provider",
        ["openai", "ollama"],
        index=0 if app_config.default_llm_type == "openai" else 1,
    )

    if llm_type == "openai":
        llm_models = ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
        default_idx = (
            llm_models.index(app_config.default_llm_model)
            if app_config.default_llm_model in llm_models
            else 0
        )
        llm_model = st.selectbox("LLM Model", llm_models, index=default_idx)
    else:
        llm_model = st.text_input("Ollama Model Name", value="llama3.2")

    embedding_type = st.selectbox(
        "Embedding Provider",
        ["openai", "chroma", "nomic"],
        index=0 if app_config.default_embedding_type == "openai" else 1,
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=app_config.default_temperature,
        step=0.1,
    )

    whisper_model = st.selectbox(
        "Whisper Model",
        ["tiny", "base", "small", "medium", "large"],
        index=1,
    )

    if st.button("🔄 Initialize", use_container_width=True):
        with st.spinner("Initializing..."):
            omni_sum, error = initialize_omni_summarizer(
                llm_type, llm_model, embedding_type, api_key, temperature, whisper_model
            )
            if error:
                st.error(f"Error: {error}")
            else:
                st.session_state.omni_summarizer = omni_sum
                st.success("Ready!")

    st.divider()

    if st.session_state.omni_summarizer:
        processed_sources = st.session_state.omni_summarizer.get_processed_sources()
        if processed_sources:
            st.subheader(f"📚 Sources ({len(processed_sources)})")
            for idx, source in enumerate(processed_sources, 1):
                with st.expander(f"{idx}. {source['title'][:40]}..."):
                    st.caption(f"**Type:** {source['type'].upper()}")
                    st.caption(f"**URL:** {source['url']}")

    st.divider()

    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.omni_summarizer:
            st.session_state.omni_summarizer.clear_context()
        st.rerun()


# Main Content
st.title("🤖 Omni Summarizer")

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show content
        if msg.get("content"):
            st.markdown(msg["content"])
        
        # Show expander if exists
        if msg.get("full_text"):
            with st.expander("📄 Full Content"):
                st.markdown(msg["full_text"][:5000] + "..." if len(msg["full_text"]) > 5000 else msg["full_text"])

# Chat input
if prompt := st.chat_input("Paste a URL or ask a question..."):
    # Auto-initialize if needed
    if not st.session_state.omni_summarizer:
        with st.spinner("Initializing models..."):
            omni_sum, error = initialize_omni_summarizer(
                llm_type, llm_model, embedding_type, api_key, temperature, whisper_model
            )
            if error:
                st.error(f"Initialization error: {error}")
                st.stop()
            else:
                st.session_state.omni_summarizer = omni_sum
    
    if st.session_state.omni_summarizer:
        # Add user message
        user_msg = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": prompt,
            "progress": None,
            "full_text": None
        }
        st.session_state.messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process message
        with st.chat_message("assistant"):
            progress_placeholder = st.empty()
            content_placeholder = st.empty()
            
            assistant_msg = {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": None,
                "progress": [],
                "full_text": None
            }
            
            try:
                for update in st.session_state.omni_summarizer.process_message(prompt):
                    status = update.get("status")
                    message = update.get("message", "")
                    
                    if status == "error":
                        assistant_msg["content"] = f"❌ {message}"
                        content_placeholder.error(message)
                        break
                    
                    elif status == "complete":
                        # Show final progress state if exists
                        if assistant_msg["progress"]:
                            with progress_placeholder.container():
                                for step in assistant_msg["progress"]:
                                    st.markdown(f"✅ {step['text']}")
                        
                        if "answer" in update:
                            # Plain text response
                            assistant_msg["content"] = update["answer"]
                            assistant_msg["progress"] = None
                            content_placeholder.markdown(update["answer"])
                        else:
                            # URL processed
                            title = update.get("title", "Content")
                            summary = update.get("summary", "")
                            
                            assistant_msg["content"] = f"### {title}\n\n{summary}\n\n✅ Added to context"
                            assistant_msg["full_text"] = update.get("full_text")
                            
                            # Display content
                            with content_placeholder.container():
                                st.markdown(f"### {title}")
                                st.markdown(summary)
                                st.success("✅ Added to context")
                                
                                if assistant_msg["full_text"]:
                                    with st.expander("📄 Full Content"):
                                        display_text = assistant_msg["full_text"][:5000]
                                        if len(assistant_msg["full_text"]) > 5000:
                                            display_text += "..."
                                        st.markdown(display_text)
                    
                    elif status == "thinking":
                        # Temporary thinking display - will be cleared
                        with progress_placeholder.container():
                            st.info(f"💭 {message}")
                    
                    else:
                        # Progress step - mark all previous as done
                        for step in assistant_msg["progress"]:
                            step["done"] = True
                        
                        # Add new step
                        assistant_msg["progress"].append({"text": message, "done": False})
                        
                        # Display progress
                        with progress_placeholder.container():
                            for step in assistant_msg["progress"]:
                                icon = "✅" if step["done"] else "⏳"
                                st.markdown(f"{icon} {step['text']}")
                
                # Mark all progress as done when complete
                if assistant_msg["progress"]:
                    for step in assistant_msg["progress"]:
                        step["done"] = True
                
                # Save message
                st.session_state.messages.append(assistant_msg)
                    
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                assistant_msg["content"] = error_msg
                assistant_msg["progress"] = None
                content_placeholder.error(error_msg)
                st.session_state.messages.append(assistant_msg)
