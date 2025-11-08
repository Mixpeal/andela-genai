import os
import re
from typing import List, Dict, Optional, Generator
from urllib.parse import urlparse
import yt_dlp
import whisper
from newspaper import Article

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage


class OmniSummarizer:
    def __init__(
        self,
        llm_type="openai",
        llm_model_name="gpt-4o-mini",
        embedding_type="openai",
        api_key: str = None,
        temperature: float = 0,
        whisper_model: str = "base",
    ):
        """Initialize the omni summarizer with LLM and embedding models"""
        self.llm_type = llm_type
        self.llm_model_name = llm_model_name
        self.embedding_type = embedding_type
        self.api_key = api_key
        self.temperature = temperature
        
        # Initialize LLM
        if llm_type == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required")
            self.llm = ChatOpenAI(
                model_name=llm_model_name,
                temperature=temperature,
                openai_api_key=api_key
            )
        elif llm_type == "ollama":
            self.llm = ChatOllama(
                model=llm_model_name,
                temperature=temperature,
                timeout=120,
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
        
        # Initialize embeddings
        if embedding_type == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=api_key,
            )
        elif embedding_type == "chroma":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings()
        elif embedding_type == "nomic":
            from langchain_community.embeddings import OllamaEmbeddings
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
        
        # Initialize Whisper for YouTube
        self.whisper_model = whisper.load_model(whisper_model)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Vector store and conversation history
        self.vector_store: Optional[Chroma] = None
        self.conversation_history: List = []
        self.processed_sources: List[Dict] = []
    
    def detect_content_type(self, text: str) -> tuple[str, Optional[str]]:
        """
        Detect if text contains a URL and what type
        Returns: (content_type, url)
        content_type: 'youtube', 'article', 'text'
        """
        # Find URLs in text
        url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
        urls = re.findall(url_pattern, text)
        
        if not urls:
            return 'text', None
        
        url = urls[0]  # Take first URL
        
        # Check if YouTube
        youtube_patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)',
        ]
        for pattern in youtube_patterns:
            if re.search(pattern, url):
                return 'youtube', url
        
        # Otherwise assume it's an article
        return 'article', url
    
    def process_youtube(self, url: str) -> Generator[Dict, None, None]:
        """
        Process YouTube video with progress updates
        Yields status updates for real-time display
        """
        try:
            os.makedirs("downloads", exist_ok=True)
            
            yield {"status": "downloading", "message": "Downloading video..."}
            
            # Download audio
            ydl_opts = {
                "format": "bestaudio/best",
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
                "outtmpl": "downloads/%(title)s.%(ext)s",
                "quiet": True,
                "no_warnings": True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_path = ydl.prepare_filename(info).replace(".webm", ".mp3")
                video_title = info.get("title", "Unknown Title")
            
            yield {"status": "transcribing", "message": f"Transcribing '{video_title}'..."}
            
            # Transcribe
            result = self.whisper_model.transcribe(audio_path)
            transcript = result["text"]
            
            yield {"status": "processing", "message": "Creating document chunks..."}
            
            # Create documents
            texts = self.text_splitter.split_text(transcript)
            documents = [
                Document(
                    page_content=chunk,
                    metadata={"source": video_title, "type": "youtube", "url": url}
                )
                for chunk in texts
            ]
            
            yield {"status": "embedding", "message": "Creating embeddings and storing in vector database..."}
            
            # Add to vector store
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name="omni_summarizer"
                )
            else:
                self.vector_store.add_documents(documents)
            
            yield {"status": "summarizing", "message": "Generating summary..."}
            
            # Generate summary
            summary = self._generate_summary(documents, video_title)
            
            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Track processed source
            self.processed_sources.append({
                "type": "youtube",
                "title": video_title,
                "url": url,
                "summary": summary
            })
            
            yield {
                "status": "complete",
                "message": "Video processed successfully!",
                "title": video_title,
                "summary": summary,
                "full_text": transcript
            }
            
        except Exception as e:
            yield {"status": "error", "message": f"Error processing video: {str(e)}"}
    
    def process_article(self, url: str) -> Generator[Dict, None, None]:
        """
        Process article with progress updates
        Yields status updates for real-time display
        """
        try:
            yield {"status": "fetching", "message": "Fetching article..."}
            
            # Fetch article
            article = Article(url)
            article.download()
            article.parse()
            
            if not article.text:
                yield {"status": "error", "message": "Could not extract article text"}
                return
            
            yield {"status": "processing", "message": f"Processing '{article.title}'..."}
            
            # Create documents
            texts = self.text_splitter.split_text(article.text)
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": article.title,
                        "type": "article",
                        "url": url,
                        "authors": ", ".join(article.authors) if article.authors else "Unknown"
                    }
                )
                for chunk in texts
            ]
            
            yield {"status": "embedding", "message": "Creating embeddings and storing in vector database..."}
            
            # Add to vector store
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name="omni_summarizer"
                )
            else:
                self.vector_store.add_documents(documents)
            
            yield {"status": "summarizing", "message": "Generating summary..."}
            
            # Generate summary
            summary = self._generate_summary(documents, article.title)
            
            # Track processed source
            self.processed_sources.append({
                "type": "article",
                "title": article.title,
                "url": url,
                "authors": article.authors,
                "summary": summary
            })
            
            yield {
                "status": "complete",
                "message": "Article processed successfully!",
                "title": article.title,
                "authors": article.authors,
                "summary": summary,
                "full_text": article.text
            }
            
        except Exception as e:
            yield {"status": "error", "message": f"Error processing article: {str(e)}"}
    
    def _generate_summary(self, documents: List[Document], title: str) -> str:
        """Generate summary from documents"""
        prompt_template = """Write a comprehensive summary of the following content titled "{title}":

{text}

Include the main points, key details, and any important conclusions.

SUMMARY:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        # Combine documents
        combined_text = "\n\n".join([doc.page_content for doc in documents[:10]])  # Limit to avoid token limits
        
        summary = chain.invoke({"title": title, "text": combined_text})
        return summary
    
    def answer_question(self, question: str) -> str:
        """
        Answer a question using RAG if vector store exists, otherwise just LLM
        """
        if self.vector_store is not None:
            # Use RAG
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            
            template = """Answer the question based on the following context and conversation history.

Context from processed sources:
{context}

Conversation history:
{history}

Question: {question}

Provide a helpful and accurate answer based on the available information. If the answer is not in the context, say so and provide a general response if possible.

Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            def format_history():
                if not self.conversation_history:
                    return "No previous conversation."
                history_text = []
                for msg in self.conversation_history[-6:]:  # Last 3 exchanges
                    role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                    history_text.append(f"{role}: {msg.content}")
                return "\n".join(history_text)
            
            rag_chain = (
                {
                    "context": retriever | format_docs,
                    "history": lambda x: format_history(),
                    "question": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            answer = rag_chain.invoke(question)
        else:
            # Just use LLM with conversation history
            messages = []
            for msg in self.conversation_history[-10:]:  # Last 5 exchanges
                messages.append(msg)
            messages.append(HumanMessage(content=question))
            
            response = self.llm.invoke(messages)
            answer = response.content
        
        # Update conversation history
        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content=answer))
        
        return answer
    
    def process_message(self, message: str) -> Generator[Dict, None, None]:
        """
        Main entry point - processes any message (URL or text)
        Yields progress updates
        """
        content_type, url = self.detect_content_type(message)
        
        if content_type == 'youtube':
            yield from self.process_youtube(url)
        elif content_type == 'article':
            yield from self.process_article(url)
        else:
            # Plain text question
            yield {"status": "thinking", "message": "Thinking..."}
            answer = self.answer_question(message)
            yield {"status": "complete", "message": "Response generated", "answer": answer}
    
    def clear_context(self):
        """Clear vector store and conversation history"""
        self.vector_store = None
        self.conversation_history = []
        self.processed_sources = []
    
    def get_processed_sources(self) -> List[Dict]:
        """Get list of all processed sources"""
        return self.processed_sources
    
    def get_model_info(self) -> Dict:
        """Return current model configuration"""
        return {
            "llm_type": self.llm_type,
            "llm_model": self.llm_model_name,
            "embedding_type": self.embedding_type,
        }

