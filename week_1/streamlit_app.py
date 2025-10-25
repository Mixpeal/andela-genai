import streamlit as st
from main import LLMApp, PROVIDER_MODELS
from appconfig import app_config
st.set_page_config(
    page_title="Andelina, personal assistant", page_icon="🚀", layout="centered"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_app" not in st.session_state:
    st.session_state.llm_app = None

if "current_provider" not in st.session_state:
    st.session_state.current_provider = None

if "current_model" not in st.session_state:
    st.session_state.current_model = None


st.title("Hi, Andelina here!")
st.markdown("""
  Chat with me!
  You can ask me anything!
""")

with st.sidebar:
    st.header("Configuration")

    provider = st.selectbox(
        "Provider",
        ["groq", "openai", "anthropic"],
        index=0,
        help="Select the LLM provider"
    )

    if provider == "groq":
        api_key = st.text_input(
            "Groq API Key", type="password", help="Enter your Groq API Key"
        )
        if not api_key:
            api_key = app_config.groq_api_key
    elif provider == "openai":
        api_key = st.text_input(
            "OpenAI API Key", type="password", help="Enter your OpenAI API Key"
        )
        if not api_key:
            api_key = app_config.openai_api_key
    else:
        api_key = st.text_input(
            "Anthropic API Key", type="password", help="Enter your Anthropic API Key"
        )
        if not api_key:
            api_key = app_config.anthropic_api_key

    model = st.selectbox(
        "Model",
        PROVIDER_MODELS[provider],
        help="Select the model to use",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Adjust the temperature of the model",
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=2048,
        value=1024,
        step=256,
        help="Adjust the maximum number of tokens to generate",
    )

    system_prompt = st.text_area(
        "System Prompt (Optional)",
        placeholder="Enter a system prompt for me",
        help="Enter a system prompt for me (Optional)",
    )

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.success("Conversation cleared")
        if st.session_state.llm_app:
            # st.session_state.llm_app.reset_conversation()
            st.success("Conversation reset")
        else:
            st.error("No conversation to reset")
        st.rerun()

if (st.session_state.llm_app is None or 
    st.session_state.current_provider != provider or 
    st.session_state.current_model != model):
    try:
        st.session_state.llm_app = LLMApp(
            api_key=api_key, 
            model_name=model, 
            provider=provider
        )
        st.session_state.current_provider = provider
        st.session_state.current_model = model
    except Exception as e:
        st.error(f"Error connecting to {provider.upper()} API: {e}")

# Display the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your message:"):
    if not api_key or not st.session_state.llm_app.api_key:
        st.warning(f"Please enter your {provider.upper()} API Key in the sidebar")

    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.llm_app.chat(
                        prompt,
                        system_prompt=system_prompt if system_prompt else None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    response = "Sorry, I encountered an error. Please try again."
