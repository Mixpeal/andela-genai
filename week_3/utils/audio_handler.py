import io
import wave
from typing import Generator, Optional

# Lazy imports to avoid import errors when audio not available
def _import_audio_deps():
    """Import audio dependencies lazily"""
    global pyaudio, webrtcvad, ElevenLabs, VoiceSettings, openai
    import pyaudio
    import webrtcvad
    from elevenlabs import ElevenLabs, VoiceSettings
    import openai


class AudioHandler:
    def __init__(
        self,
        elevenlabs_api_key: str,
        openai_api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Default ElevenLabs voice
        vad_aggressiveness: int = 3,  # 0-3, higher = more aggressive
        language: str = "en",  # Language code for transcription
        tts_provider: str = "openai"  # TTS provider: "elevenlabs" or "openai"
    ):
        """Initialize audio handler with ElevenLabs and VAD"""
        # Import dependencies
        _import_audio_deps()
        
        self.tts_provider = tts_provider
        self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key) if elevenlabs_api_key else None
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        self.language = language
        
        self.voice_id = voice_id
        
        # Audio settings
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        
        # VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # State
        self.is_listening = False
        self.is_speaking = False
    
    def start_listening(self):
        """Start audio input stream"""
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size
        )
        self.is_listening = True
    
    def stop_listening(self):
        """Stop audio input stream"""
        self.is_listening = False
        try:
            if self.stream:
                try:
                    self.stream.stop_stream()
                except:
                    pass
                try:
                    self.stream.close()
                except:
                    pass
                self.stream = None
        except Exception as e:
            print(f"Error stopping stream: {e}")
    
    def listen_with_vad(self, silence_duration: float = 1.0) -> Optional[bytes]:
        """
        Listen for speech and return audio when silence detected
        silence_duration: seconds of silence before stopping
        """
        if not self.is_listening:
            self.start_listening()
        
        frames = []
        silence_frames = int(silence_duration * 1000 / self.frame_duration)
        consecutive_silence = 0
        speech_started = False
        
        while self.is_listening:
            frame = self.stream.read(self.frame_size, exception_on_overflow=False)
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            
            if is_speech:
                speech_started = True
                consecutive_silence = 0
                frames.append(frame)
            elif speech_started:
                frames.append(frame)
                consecutive_silence += 1
                
                if consecutive_silence >= silence_frames:
                    # End of speech detected
                    break
        
        if not frames:
            return None
        
        # Convert frames to bytes
        audio_data = b''.join(frames)
        return audio_data
    
    def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio using OpenAI Whisper API"""
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data)
        
        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"
        
        # Transcribe
        try:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=wav_buffer,
                language=self.language  # Specify language
            )
            return response.text
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech using selected provider
        Returns audio bytes
        """
        if self.tts_provider == "elevenlabs":
            return self._elevenlabs_tts(text)
        elif self.tts_provider == "openai":
            return self._openai_tts(text)
        else:
            print(f"Unknown TTS provider: {self.tts_provider}")
            return b""
    
    def _elevenlabs_tts(self, text: str) -> bytes:
        """ElevenLabs TTS"""
        try:
            if not self.elevenlabs_client:
                raise Exception("ElevenLabs API key not provided")
                
            audio_generator = self.elevenlabs_client.text_to_speech.convert(
                voice_id=self.voice_id,
                text=text,
                model_id="eleven_turbo_v2"
            )
            
            # Convert generator to bytes
            audio_bytes = b"".join(audio_generator)
            return audio_bytes
                    
        except Exception as e:
            print(f"ElevenLabs TTS error: {e}")
            return b""
    
    def _openai_tts(self, text: str) -> bytes:
        """OpenAI TTS - returns WAV format"""
        try:
            import io
            from pydub import AudioSegment
            
            response = openai.audio.speech.create(
                model="tts-1",  # or "tts-1-hd" for higher quality
                voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
                input=text,
                response_format="mp3"
            )
            
            # Convert MP3 to WAV for playback
            mp3_data = response.content
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            
            # Export as WAV
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            
            return wav_buffer.read()
                    
        except Exception as e:
            print(f"OpenAI TTS error: {e}")
            return b""
    
    
    def play_audio(self, audio_bytes: bytes):
        """Play audio from bytes (WAV format)"""
        if not audio_bytes:
            return
            
        self.is_speaking = True
        
        try:
            import wave
            import io
            
            # Read WAV file to get parameters
            wav_buffer = io.BytesIO(audio_bytes)
            with wave.open(wav_buffer, 'rb') as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            
            # Open output stream with correct parameters
            output_stream = self.audio.open(
                format=self.audio.get_format_from_width(sample_width),
                channels=channels,
                rate=framerate,
                output=True
            )
            
            # Play audio in chunks
            chunk_size = 1024
            for i in range(0, len(frames), chunk_size):
                if not self.is_speaking:
                    break
                chunk = frames[i:i + chunk_size]
                output_stream.write(chunk)
                
        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            if 'output_stream' in locals():
                output_stream.stop_stream()
                output_stream.close()
            self.is_speaking = False
    
    def stop_speaking(self):
        """Interrupt current speech"""
        self.is_speaking = False
    
    def cleanup(self):
        """Clean up audio resources"""
        try:
            self.stop_listening()
        except:
            pass
        try:
            self.stop_speaking()
        except:
            pass
        try:
            if self.audio:
                self.audio.terminate()
        except:
            pass
    
    def get_available_voices(self):
        """Get list of available ElevenLabs voices"""
        try:
            voices = self.elevenlabs_client.voices.get_all()
            return [(voice.voice_id, voice.name) for voice in voices.voices]
        except Exception as e:
            print(f"Error getting voices: {e}")
            return []

