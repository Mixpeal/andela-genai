import os
import re
from typing import Dict, Generator, List
import yt_dlp
import whisper
from newspaper import Article


class ContentExtractor:
    """Extract content from URLs (YouTube videos and articles)"""
    
    def __init__(self, whisper_model: str = "base"):
        """Initialize content extractor"""
        self.whisper_model = whisper.load_model(whisper_model)
    
    def detect_all_urls(self, text: str) -> List[Dict[str, str]]:
        """
        Detect all URLs in text and classify them
        Returns: List of dicts with 'type' and 'url' keys
        """
        # Find all URLs in text
        url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
        urls = re.findall(url_pattern, text)
        
        if not urls:
            return []
        
        classified_urls = []
        
        for url in urls:
            # Check if YouTube
            youtube_patterns = [
                r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)',
            ]
            is_youtube = False
            for pattern in youtube_patterns:
                if re.search(pattern, url):
                    classified_urls.append({'type': 'youtube', 'url': url})
                    is_youtube = True
                    break
            
            # If not YouTube, assume it's an article
            if not is_youtube:
                classified_urls.append({'type': 'article', 'url': url})
        
        return classified_urls
    
    def get_plain_text(self, text: str) -> str:
        """Extract plain text without URLs"""
        url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
        plain_text = re.sub(url_pattern, '', text).strip()
        return plain_text
    
    def extract_youtube_content(self, url: str) -> Generator[Dict, None, None]:
        """
        Extract content from YouTube video with progress updates
        Yields status updates for real-time display
        """
        try:
            os.makedirs("downloads", exist_ok=True)
            
            yield {"status": "downloading", "message": "📥 Downloading YouTube video..."}
            
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
                video_description = info.get("description", "")
            
            yield {"status": "transcribing", "message": f"📝 Transcribing '{video_title}'..."}
            
            # Transcribe
            result = self.whisper_model.transcribe(audio_path)
            transcript = result["text"]
            
            # Clean up audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            yield {
                "status": "complete",
                "message": "✅ YouTube content extracted successfully!",
                "content_type": "youtube",
                "title": video_title,
                "content": transcript,
                "description": video_description,
                "url": url
            }
            
        except Exception as e:
            yield {"status": "error", "message": f"Error extracting YouTube content: {str(e)}"}
    
    def extract_article_content(self, url: str) -> Generator[Dict, None, None]:
        """
        Extract content from article with progress updates
        Yields status updates for real-time display
        """
        try:
            yield {"status": "fetching", "message": "📄 Fetching article..."}
            
            # Fetch article
            article = Article(url)
            article.download()
            article.parse()
            
            if not article.text:
                yield {"status": "error", "message": "Could not extract article text"}
                return
            
            yield {
                "status": "complete",
                "message": "✅ Article content extracted successfully!",
                "content_type": "article",
                "title": article.title,
                "content": article.text,
                "authors": article.authors,
                "url": url
            }
            
        except Exception as e:
            yield {"status": "error", "message": f"Error extracting article content: {str(e)}"}
    
    def extract_all_content(self, text: str) -> Generator[Dict, None, None]:
        """
        Extract content from all URLs in text
        Yields progress updates and returns all extracted content
        """
        urls = self.detect_all_urls(text)
        plain_text = self.get_plain_text(text)
        
        if not urls:
            # No URLs found, just plain text
            yield {
                "status": "complete",
                "content_type": "text",
                "plain_text": plain_text,
                "sources": []
            }
            return
        
        # Extract content from all URLs
        extracted_sources = []
        
        for idx, url_info in enumerate(urls, 1):
            url_type = url_info['type']
            url = url_info['url']
            
            yield {
                "status": "processing", 
                "message": f"📌 Processing {url_type.upper()} {idx}/{len(urls)}..."
            }
            
            if url_type == 'youtube':
                for update in self.extract_youtube_content(url):
                    if update.get("status") == "complete":
                        extracted_sources.append(update)
                    else:
                        yield update
            
            elif url_type == 'article':
                for update in self.extract_article_content(url):
                    if update.get("status") == "complete":
                        extracted_sources.append(update)
                    else:
                        yield update
        
        # Return all extracted content
        yield {
            "status": "complete",
            "content_type": "multiple" if len(extracted_sources) > 1 else extracted_sources[0].get("content_type"),
            "plain_text": plain_text,
            "sources": extracted_sources,
            "total_sources": len(extracted_sources)
        }

