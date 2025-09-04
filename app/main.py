import logging
import time
import io
import wave
import numpy as np
import tempfile
import os
import re
import unicodedata
from make87_messages.audio.frame_pcm_s16le_pb2 import FramePcmS16le
from make87.encodings import ProtobufEncoder
from make87.interfaces.zenoh import ZenohInterface

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    raise ImportError("pyttsx3 is required for offline TTS. Install: pip install pyttsx3")

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    format="[%(asctime)sZ %(levelname)s  %(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%dT%H:%M:%S"
)

logger = logging.getLogger(__name__)


class TextToFramePcm:
    """Converts text to FramePcmS16le format using offline pyttsx3/eSpeak-NG"""

    def __init__(self, voice_id: str = None, rate: int = 150, volume: float = 1.0):
        """
        Initialize the offline text-to-speech converter

        Args:
            voice_id: Specific voice ID for pyttsx3 (None = auto-select best voice)
            rate: Speech rate in words per minute (default: 150)
            volume: Speech volume 0.0-1.0 (default: 1.0)
        """
        self.voice_id = voice_id
        self.rate = rate
        self.volume = volume
        
        if not PYTTSX3_AVAILABLE:
            raise RuntimeError("pyttsx3 not available. Install: pip install pyttsx3")
            
        self._init_pyttsx3()
        logger.info("Initialized offline TTS with pyttsx3/eSpeak-NG (using native sample rates)")

    @classmethod  
    def list_available_voices(cls):
        """List available pyttsx3 voices"""
        if not PYTTSX3_AVAILABLE:
            print("pyttsx3 not available")
            return []
            
        try:
            import pyttsx3
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            
            if not voices:
                print("No voices found")
                return []
                
            print("Available offline TTS voices:")
            print("=" * 50)
            voice_info = []
            for i, voice in enumerate(voices):
                info = {
                    'index': i,
                    'id': voice.id,
                    'name': getattr(voice, 'name', 'Unknown'),
                    'age': getattr(voice, 'age', 'Unknown'),
                    'gender': getattr(voice, 'gender', 'Unknown'),
                    'languages': getattr(voice, 'languages', [])
                }
                voice_info.append(info)
                
                # Try to detect if it's male/female for pitch indication
                gender = info['gender'].lower() if info['gender'] != 'Unknown' else ''
                name = info['name'].lower()
                
                pitch_hint = ""
                if 'male' in gender and 'female' not in gender:
                    pitch_hint = " (Lower pitch)"
                elif 'female' in gender:
                    pitch_hint = " (Higher pitch)"
                elif any(male_name in name for male_name in ['david', 'james', 'mark', 'paul', 'michael']):
                    pitch_hint = " (Likely lower pitch)"
                elif any(female_name in name for female_name in ['zira', 'hazel', 'susan', 'mary', 'linda']):
                    pitch_hint = " (Likely higher pitch)"
                
                print(f"[{i}] {info['name']}{pitch_hint}")
                print(f"    ID: {info['id']}")
                print(f"    Gender: {info['gender']}")
                print(f"    Age: {info['age']}")
                if info['languages']:
                    print(f"    Languages: {info['languages']}")
                print()
                
            engine.stop()
            return voice_info
            
        except Exception as e:
            print(f"Error listing voices: {e}")
            return []

    def _init_pyttsx3(self):
        """Initialize pyttsx3 with voice selection for better quality"""
        try:
            self.engine = pyttsx3.init()
            
            # Set basic properties
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            logger.info("pyttsx3 engine initialized successfully")
            
            # Configure voice selection
            try:
                voices = self.engine.getProperty('voices')
                logger.info(f"Found {len(voices) if voices else 0} voices")
                
                if not voices:
                    logger.warning("No voices found, using engine defaults")
                    return
                    
                # Use specific voice if provided
                if self.voice_id:
                    for voice in voices:
                        if hasattr(voice, 'id') and voice.id == self.voice_id:
                            self.engine.setProperty('voice', voice.id)
                            logger.info(f"Set specific voice: {voice.id}")
                            return
                    logger.warning(f"Requested voice ID '{self.voice_id}' not found, using auto-selection")
                
                # Auto-select best voice (prefer male/lower pitch voices)
                male_voices = []
                female_voices = []
                other_voices = []
                
                for voice in voices:
                    if not hasattr(voice, 'id'):
                        continue
                        
                    voice_id = voice.id.lower()
                    name = getattr(voice, 'name', '').lower()
                    gender = getattr(voice, 'gender', '').lower()
                    
                    # Check if it's English
                    is_english = ('en' in voice_id or 'english' in voice_id or 
                                'us' in voice_id or 'uk' in voice_id or 'gb' in voice_id)
                    
                    if not is_english:
                        continue
                        
                    # Try to detect gender
                    is_male = ('male' in gender and 'female' not in gender) or \
                             any(male_name in name for male_name in ['david', 'james', 'mark', 'paul', 'michael', 'daniel'])
                    is_female = 'female' in gender or \
                               any(female_name in name for female_name in ['zira', 'hazel', 'susan', 'mary', 'linda', 'cortana'])
                    
                    if is_male:
                        male_voices.append(voice)
                    elif is_female:
                        female_voices.append(voice)
                    else:
                        other_voices.append(voice)
                
                # Prefer male voices (typically lower pitch), then others, then female
                preferred_voices = male_voices + other_voices + female_voices
                
                if preferred_voices:
                    selected_voice = preferred_voices[0]
                    self.engine.setProperty('voice', selected_voice.id)
                    
                    voice_type = ""
                    if selected_voice in male_voices:
                        voice_type = " (male - lower pitch)"
                    elif selected_voice in female_voices:
                        voice_type = " (female - higher pitch)"
                    else:
                        voice_type = " (unknown gender)"
                        
                    logger.info(f"Selected voice: {getattr(selected_voice, 'name', selected_voice.id)}{voice_type}")
                else:
                    # Use first available voice as fallback
                    self.engine.setProperty('voice', voices[0].id)
                    logger.info(f"Using fallback voice: {getattr(voices[0], 'name', voices[0].id)}")
                    
            except Exception as e:
                logger.warning(f"Voice selection failed, using default: {e}")
                
        except Exception as e:
            logger.error(f"pyttsx3 initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize offline TTS: {e}")

    @staticmethod
    def clean_text_for_tts(text: str) -> str:
        """
        Clean text for TTS by removing emojis and problematic characters
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text suitable for TTS
        """
        if not text or not text.strip():
            return ""
            
        # Remove emojis using Unicode categories
        # Emojis are typically in categories: So (Symbol, other) and some Sm (Symbol, math)
        cleaned = ''.join(char for char in text if unicodedata.category(char) not in ['So', 'Cn'])
        
        # Additional emoji patterns that might be missed
        emoji_patterns = [
            r'[\U0001F600-\U0001F64F]',  # emoticons
            r'[\U0001F300-\U0001F5FF]',  # symbols & pictographs
            r'[\U0001F680-\U0001F6FF]',  # transport & map symbols
            r'[\U0001F1E0-\U0001F1FF]',  # flags (iOS)
            r'[\U00002700-\U000027BF]',  # dingbats
            r'[\U0001F900-\U0001F9FF]',  # supplemental symbols
            r'[\U00002600-\U000026FF]',  # miscellaneous symbols
            r'[\U00002B00-\U00002BFF]',  # miscellaneous symbols and arrows
        ]
        
        for pattern in emoji_patterns:
            cleaned = re.sub(pattern, '', cleaned)
        
        # Remove other problematic characters for TTS (just remove, don't replace)
        problematic_chars = [
            '\u200b',      # zero-width space
            '\u200c',      # zero-width non-joiner
            '\u200d',      # zero-width joiner
            '\ufeff',      # byte order mark
            '\u2028',      # line separator → space
            '\u2029',      # paragraph separator → space
            # Remove common symbols that TTS engines struggle with
            '♥', '♡', '★', '☆', '✓', '✔', '✗', '✘',
            '❤', '💙', '💚', '💛', '💜', '🖤', '🤍', '🤎',
        ]
        
        # Replace line/paragraph separators with spaces, others with empty string
        for char in problematic_chars:
            if char in ['\u2028', '\u2029']:
                cleaned = cleaned.replace(char, ' ')
            else:
                cleaned = cleaned.replace(char, '')
        
        # Clean up whitespace
        cleaned = ' '.join(cleaned.split())  # Replace multiple spaces with single space
        cleaned = cleaned.strip()
        
        # Basic validation
        if not cleaned:
            logger.warning("Text became empty after cleaning")
            return "Empty message."
        
        # Truncate very long messages to avoid TTS timeouts
        max_length = 1000  # Adjust based on your needs
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length].rsplit(' ', 1)[0] + "..."
            logger.warning(f"Text truncated to {max_length} characters")
        
        return cleaned

    def text_to_wav_bytes(self, text: str) -> bytes:
        """
        Convert text to WAV format bytes using pyttsx3

        Args:
            text: Text to convert to speech

        Returns:
            WAV audio data as bytes
        """
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file:
            tmp_filename = tmp_file.name
            
            self.engine.save_to_file(text, tmp_filename)
            self.engine.runAndWait()

            # Read immediately and return
            with open(tmp_filename, 'rb') as f:
                wav_data = f.read()
            
            # File is automatically deleted when leaving the context
            return wav_data

    def wav_to_pcm_s16le(self, wav_data: bytes) -> tuple[bytes, int, int]:
        """
        Convert WAV data to PCM S16LE format

        Args:
            wav_data: WAV format audio data

        Returns:
            Tuple of (pcm_data, sample_rate, channels)
        """
        wav_io = io.BytesIO(wav_data)

        with wave.open(wav_io, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()

            # Convert to numpy array for processing
            if sample_width == 1:
                audio_data = np.frombuffer(frames, dtype=np.uint8)
                audio_data = ((audio_data.astype(np.float32) - 128) * 256).astype(np.int16)
            elif sample_width == 2:
                audio_data = np.frombuffer(frames, dtype=np.int16)
            elif sample_width == 4:
                audio_data = np.frombuffer(frames, dtype=np.int32)
                audio_data = (audio_data // 65536).astype(np.int16)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            # Ensure little-endian format
            pcm_data = audio_data.astype('<i2').tobytes()
            return pcm_data, sample_rate, channels

    def text_to_frame_pcm_s16le(self, text: str, pts: int = 0, force_mono: bool = True) -> FramePcmS16le:
        """
        Convert text to FramePcmS16le format using native sample rates

        Args:
            text: Text to convert to speech
            pts: Presentation timestamp
            force_mono: Force conversion to mono audio

        Returns:
            FramePcmS16le protobuf message
        """
        # Clean text for TTS processing
        cleaned_text = self.clean_text_for_tts(text)
        
        if not cleaned_text:
            logger.warning("No valid text to convert after cleaning")
            # Return minimal silent frame - will use whatever sample rate eSpeak would use
            # Generate a tiny bit of audio to get the native sample rate
            try:
                test_wav = self.text_to_wav_bytes(".")
                _, native_sr, _ = self.wav_to_pcm_s16le(test_wav)
            except:
                native_sr = 22050  # fallback
                
            silent_duration = 0.1  # 100ms of silence
            samples = int(silent_duration * native_sr)
            pcm_data = np.zeros(samples, dtype=np.int16).tobytes()
            
            return FramePcmS16le(
                data=pcm_data,
                pts=pts,
                time_base=FramePcmS16le.Fraction(num=1, den=native_sr),
                channels=1
            )
        
        logger.debug(f"Converting text to speech: '{cleaned_text[:50]}{'...' if len(cleaned_text) > 50 else ''}'")
        
        # Convert using pyttsx3 and get native sample rate
        try:
            wav_data = self.text_to_wav_bytes(cleaned_text)
            pcm_data, native_sr, channels = self.wav_to_pcm_s16le(wav_data)
            logger.debug(f"Generated audio: {native_sr}Hz, {channels}ch, {len(pcm_data)} bytes")
            
        except Exception as e:
            logger.error(f"TTS conversion failed: {e}")
            # Return silent frame on error
            native_sr = 22050  # fallback
            silent_duration = 0.1
            samples = int(silent_duration * native_sr)
            pcm_data = np.zeros(samples, dtype=np.int16).tobytes()
            channels = 1

        # Handle stereo to mono conversion if needed
        if force_mono and channels == 2:
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            stereo_array = audio_array.reshape(-1, 2)
            mono_array = np.mean(stereo_array, axis=1).astype(np.int16)
            pcm_data = mono_array.tobytes()
            channels = 1

        # NO RESAMPLING - just use native sample rate from eSpeak

        # Create the FramePcmS16le protobuf message with native sample rate
        frame = FramePcmS16le(
            data=pcm_data,
            pts=pts,
            time_base=FramePcmS16le.Fraction(num=1, den=native_sr),
            channels=channels
        )
        
        duration_seconds = len(pcm_data) / (2 * channels * native_sr)
        logger.debug(f"Generated frame: {native_sr}Hz, {channels}ch, {len(pcm_data)} bytes, {duration_seconds:.2f}s")

        return frame


def main():
    logger.info("Starting offline TTS service...")
    
    message_encoder = ProtobufEncoder(message_type=FramePcmS16le)
    zenoh_interface = ZenohInterface(name="zenoh")

    publisher = zenoh_interface.get_publisher("tts_audio")
    subscriber = zenoh_interface.get_subscriber("tts_text")

    # Initialize converter with native sample rate handling
    try:
        converter = TextToFramePcm(
            # VOICE OPTIONS:
            voice_id="gmw/en",        # Use specific eSpeak voice
            # voice_id=None,          # Auto-select best voice
            
            # SPEECH SETTINGS:
            rate=140,                 # Words per minute (100-200 typical)
            volume=1.0                # Volume 0.0-1.0
        )
        logger.info("Offline TTS service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize TTS service: {e}")
        raise

    logger.info("Listening for text messages...")
    
    for msg in subscriber:
        try:
            text = msg.payload.to_bytes().decode("utf-8")
            logger.info(f"Received text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            
            start_time = time.time()
            frame = converter.text_to_frame_pcm_s16le(text, pts=int(time.time() * 1000))
            conversion_time = time.time() - start_time
            
            message_encoded = message_encoder.encode(frame)
            publisher.put(payload=message_encoded)

            # Log the actual sample rate being used
            actual_sr = frame.time_base.den
            logger.info(f"Published offline TTS audio at {actual_sr}Hz ({conversion_time:.2f}s conversion time)")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            continue


if __name__ == "__main__":
    # Uncomment to see available voices:
    # print("=== Available Voices ===")
    # TextToFramePcm.list_available_voices()
    # print("========================")
    
    main()