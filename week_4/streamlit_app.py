import streamlit as st
import uuid
from utils import BlogGenerator, ChatClassifier
from appconfig import app_config

st.set_page_config(
    page_title="Blog Generator", page_icon="✍️", layout="wide"
)

# Initialize session state
if "blog_generator" not in st.session_state:
    st.session_state.blog_generator = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "generated_blogs" not in st.session_state:
    st.session_state.generated_blogs = []

if "auto_initialized" not in st.session_state:
    st.session_state.auto_initialized = False

if "chat_classifier" not in st.session_state:
    st.session_state.chat_classifier = None


def initialize_blog_generator(api_key, model_name, temperature):
    """Initialize blog generator"""
    try:
        generator = BlogGenerator(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature
        )
        return generator, None
    except Exception as e:
        return None, str(e)


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key
    st.subheader("OpenAI Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=app_config.openai_api_key or "",
        help="Your OpenAI API key"
    )
    if not api_key:
        api_key = app_config.openai_api_key
    
    st.divider()
    
    # Model Settings
    st.subheader("Model Settings")
    model_name = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Select the OpenAI model to use"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=app_config.default_temperature,
        step=0.1,
        help="Higher values make output more creative, lower values more focused"
    )
    
    st.divider()
    
    # Initialize button (manual override)
    if st.button("🔄 Re-initialize", use_container_width=True):
        with st.spinner("Re-initializing blog generator..."):
            generator, error = initialize_blog_generator(api_key, model_name, temperature)
            if error:
                st.error(f"Error: {error}")
            else:
                st.session_state.blog_generator = generator
                st.session_state.auto_initialized = True
                st.success("✅ Ready to generate blogs!")
                st.rerun()
    
    # Show initialization status
    if st.session_state.blog_generator:
        st.success("✅ Generator Ready")
    
    st.divider()
    
    # Generated blogs history
    if st.session_state.generated_blogs:
        st.subheader(f"📚 Generated Blogs ({len(st.session_state.generated_blogs)})")
        for idx, blog_data in enumerate(st.session_state.generated_blogs, 1):
            with st.expander(f"{idx}. {blog_data['blog'].title[:50]}..."):
                st.caption(f"**Topic:** {blog_data['topic']}")
                st.caption(f"**Words:** ~{len(blog_data['blog'].content.split())} words")
    
    st.divider()
    
    # Clear button
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.generated_blogs = []
        if st.session_state.blog_generator:
            st.session_state.blog_generator.clear_history()
        st.rerun()


# Auto-initialize on first load
if not st.session_state.auto_initialized and api_key:
    with st.spinner("Initializing blog generator..."):
        generator, error = initialize_blog_generator(api_key, model_name, temperature)
        if not error:
            st.session_state.blog_generator = generator
            st.session_state.chat_classifier = ChatClassifier(generator.llm)
            st.session_state.auto_initialized = True

# Main Content
st.title("✍️ AI Blog Generator")
st.markdown("Generate high-quality, SEO-optimized blog posts from any topic using LangGraph and OpenAI")

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show content
        if msg.get("content"):
            st.markdown(msg["content"])
        
        # Show generated blog if exists
        if msg.get("blog"):
            blog = msg["blog"]
            extracted = msg.get("extracted_content")
            
            st.divider()
            
            # Show source information if content was extracted
            if extracted:
                sources = extracted.get("sources", [])
                if sources:
                    if len(sources) == 1:
                        source = sources[0]
                        content_type = source.get("content_type")
                        if content_type == "youtube":
                            st.info(f"📹 **Source:** YouTube Video - [{source.get('title')}]({source.get('url')})")
                        elif content_type == "article":
                            authors = source.get('authors', [])
                            author_text = f" by {', '.join(authors)}" if authors else ""
                            st.info(f"📄 **Source:** Article - [{source.get('title')}{author_text}]({source.get('url')})")
                    else:
                        st.info(f"📚 **Sources:** {len(sources)} sources used")
                        with st.expander("View all sources"):
                            for idx, source in enumerate(sources, 1):
                                content_type = source.get("content_type")
                                if content_type == "youtube":
                                    st.markdown(f"{idx}. 📹 YouTube: [{source.get('title')}]({source.get('url')})")
                                elif content_type == "article":
                                    authors = source.get('authors', [])
                                    author_text = f" by {', '.join(authors)}" if authors else ""
                                    st.markdown(f"{idx}. 📄 Article: [{source.get('title')}{author_text}]({source.get('url')})")
            
            # Title
            st.markdown(f"# {blog.title}")
            
            # Keywords in a nice format
            if blog.seo_keywords:
                st.markdown("**🔍 SEO Keywords:**")
                keywords = [k.strip() for k in blog.seo_keywords.split(",")]
                st.write(", ".join([f"`{k}`" for k in keywords[:10]]))  # Show first 10
            
            st.divider()
            
            # Tabs for outline and content
            tab1, tab2 = st.tabs(["📝 Blog Content", "📋 Outline"])
            
            with tab1:
                st.markdown(blog.content)
            
            with tab2:
                st.markdown(blog.outline)
            
            st.divider()
            
            # Download button
            markdown_content = st.session_state.blog_generator.export_blog_markdown(blog)
            st.download_button(
                label="📥 Download as Markdown",
                data=markdown_content,
                file_name=f"{blog.title[:30].replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )


# Chat input
if prompt := st.chat_input("Enter a topic, or paste URL(s) for YouTube videos or articles..."):
    # Auto-initialize if needed
    if not st.session_state.blog_generator:
        if not api_key:
            st.error("⚠️ Please provide an OpenAI API key in the sidebar")
            st.stop()
        
        with st.spinner("Initializing blog generator..."):
            generator, error = initialize_blog_generator(api_key, model_name, temperature)
            if error:
                st.error(f"Initialization error: {error}")
                st.stop()
            else:
                st.session_state.blog_generator = generator
    
    if st.session_state.blog_generator:
        # Classify user intent
        has_blogs = len(st.session_state.generated_blogs) > 0
        
        with st.spinner("🤔 Understanding your request..."):
            intent = st.session_state.chat_classifier.classify_message(prompt, has_blogs)
        
        # Add user message
        user_msg = {
            "id": str(uuid.uuid4()),
            "role": "user",
            "content": prompt,
            "blog": None,
            "extracted_content": None
        }
        st.session_state.messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Handle based on intent
        if intent.intent == "chat" or intent.intent == "question_about_blog":
            # Chat response (streaming)
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                assistant_msg = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": "",
                    "blog": None,
                    "extracted_content": None
                }
                
                try:
                    # Stream the chat response
                    response_text = ""
                    for chunk in st.session_state.blog_generator.chat(prompt):
                        response_text += chunk
                        response_placeholder.markdown(response_text + "▌")
                    
                    # Final display without cursor
                    response_placeholder.markdown(response_text)
                    assistant_msg["content"] = response_text
                    
                    # Save message
                    st.session_state.messages.append(assistant_msg)
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    response_placeholder.error(error_msg)
                    assistant_msg["content"] = error_msg
                    st.session_state.messages.append(assistant_msg)
        
        else:  # generate_blog
            # Generate blog
            with st.chat_message("assistant"):
                progress_placeholder = st.empty()
                content_placeholder = st.empty()
                
                assistant_msg = {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": None,
                    "blog": None,
                    "extracted_content": None
                }
                
                try:
                    for update in st.session_state.blog_generator.generate_blog(prompt):
                        status = update.get("status")
                        message = update.get("message", "")
                        
                        if status == "error":
                            assistant_msg["content"] = f"❌ {message}"
                            content_placeholder.error(message)
                            break
                        
                        elif status == "complete":
                            blog = update.get("blog")
                            extracted = update.get("extracted_content")
                            assistant_msg["blog"] = blog
                            assistant_msg["extracted_content"] = extracted
                            assistant_msg["content"] = "✅ Successfully generated blog post!"
                            
                            # Store in history
                            st.session_state.generated_blogs.append({
                                "topic": prompt,
                                "blog": blog
                            })
                            
                            # Display success message
                            progress_placeholder.success("✅ Blog generated successfully!")
                            
                            # Display the blog
                            with content_placeholder.container():
                                st.divider()
                                
                                # Show source information if content was extracted
                                if extracted:
                                    sources = extracted.get("sources", [])
                                    if sources:
                                        if len(sources) == 1:
                                            source = sources[0]
                                            content_type = source.get("content_type")
                                            if content_type == "youtube":
                                                st.info(f"📹 **Source:** YouTube Video - [{source.get('title')}]({source.get('url')})")
                                            elif content_type == "article":
                                                authors = source.get('authors', [])
                                                author_text = f" by {', '.join(authors)}" if authors else ""
                                                st.info(f"📄 **Source:** Article - [{source.get('title')}{author_text}]({source.get('url')})")
                                        else:
                                            st.info(f"📚 **Sources:** {len(sources)} sources used")
                                            with st.expander("View all sources"):
                                                for idx, source in enumerate(sources, 1):
                                                    content_type = source.get("content_type")
                                                    if content_type == "youtube":
                                                        st.markdown(f"{idx}. 📹 YouTube: [{source.get('title')}]({source.get('url')})")
                                                    elif content_type == "article":
                                                        authors = source.get('authors', [])
                                                        author_text = f" by {', '.join(authors)}" if authors else ""
                                                        st.markdown(f"{idx}. 📄 Article: [{source.get('title')}{author_text}]({source.get('url')})")
                                
                                # Title
                                st.markdown(f"# {blog.title}")
                                
                                # Keywords
                                if blog.seo_keywords:
                                    st.markdown("**🔍 SEO Keywords:**")
                                    keywords = [k.strip() for k in blog.seo_keywords.split(",")]
                                    st.write(", ".join([f"`{k}`" for k in keywords[:10]]))
                                
                                st.divider()
                                
                                # Tabs
                                tab1, tab2 = st.tabs(["📝 Blog Content", "📋 Outline"])
                                
                                with tab1:
                                    st.markdown(blog.content)
                                
                                with tab2:
                                    st.markdown(blog.outline)
                                
                                st.divider()
                                
                                # Download button
                                markdown_content = st.session_state.blog_generator.export_blog_markdown(blog)
                                st.download_button(
                                    label="📥 Download as Markdown",
                                    data=markdown_content,
                                    file_name=f"{blog.title[:30].replace(' ', '_')}.md",
                                    mime="text/markdown",
                                    use_container_width=True,
                                    key=f"download_{assistant_msg['id']}"
                                                )
                        
                        else:
                            progress_placeholder.info(f"{message}")
                    
                    # Save message
                    st.session_state.messages.append(assistant_msg)
                        
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    assistant_msg["content"] = error_msg
                    content_placeholder.error(error_msg)
                    st.session_state.messages.append(assistant_msg)

# Footer
st.divider()
st.caption("✍️ AI Blog Generator powered by LangGraph and OpenAI")

