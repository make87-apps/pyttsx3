import logging
import time
import io
import wave
import numpy as np
import tempfile
import os
from make87_messages.audio.frame_pcm_s16le_pb2 import FramePcmS16le
from make87.encodings import ProtobufEncoder
from make87.interfaces.zenoh import ZenohInterface

try:
    from gtts import gTTS
    from pydub import AudioSegment
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    format="[%(asctime)sZ %(levelname)s  %(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%dT%H:%M:%S"
)

logger = logging.getLogger(__name__)


class TextToFramePcm:
    """Converts text to FramePcmS16le format using gTTS (preferred) or pyttsx3 (fallback)"""

    def __init__(self, sample_rate: int = 16000, prefer_gtts: bool = True, language: str = 'en', 
                 tld: str = 'com', slow: bool = False):
        """
        Initialize the text-to-speech converter

        Args:
            sample_rate: Audio sample rate (default: 16000 Hz)
            prefer_gtts: Use Google TTS if available (recommended)
            language: Language for gTTS (default: 'en')
            tld: Top-level domain for gTTS voice variety (default: 'com')
            slow: Slow speech for gTTS (default: False)
        """
        self.sample_rate = sample_rate
        self.language = language
        self.tld = tld
        self.slow = slow
        
        # Determine which TTS engine to use
        if prefer_gtts and GTTS_AVAILABLE:
            self.use_gtts = True
            logger.info("Initialized with Google TTS")
        elif PYTTSX3_AVAILABLE:
            self.use_gtts = False
            self._init_pyttsx3()
            logger.info("Initialized with pyttsx3")
        else:
            raise RuntimeError("No TTS engine available. Install: pip install gtts pydub")

    def _init_pyttsx3(self):
        """Initialize pyttsx3 with robust error handling"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 1.0)
            
            # Try to configure voice safely
            try:
                voices = self.engine.getProperty('voices')
                if voices and len(voices) > 0:
                    # Find a working English voice
                    for voice in voices:
                        try:
                            if hasattr(voice, 'id') and 'en' in voice.id.lower():
                                self.engine.setProperty('voice', voice.id)
                                logger.info(f"Set voice to: {voice.id}")
                                break
                        except Exception:
                            continue
            except Exception as e:
                logger.warning(f"Voice configuration failed, using defaults: {e}")
                
        except Exception as e:
            logger.error(f"pyttsx3 initialization failed: {e}")
            raise RuntimeError(f"pyttsx3 failed to initialize: {e}")

    def text_to_wav_bytes(self, text: str) -> bytes:
        """
        Convert text to WAV format bytes

        Args:
            text: Text to convert to speech

        Returns:
            WAV audio data as bytes
        """
        if self.use_gtts:
            return self._gtts_to_wav(text)
        else:
            return self._pyttsx3_to_wav(text)

    def _gtts_to_wav(self, text: str) -> bytes:
        """Convert text to WAV using Google TTS"""
        try:
            # Create TTS object with configurable parameters
            tts = gTTS(text=text, lang=self.language, tld=self.tld, slow=self.slow)
            
            # Save to temporary MP3 file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
                tmp_mp3_filename = tmp_mp3.name
                
            try:
                tts.save(tmp_mp3_filename)
                
                # Try to convert MP3 to WAV using pydub
                try:
                    audio = AudioSegment.from_mp3(tmp_mp3_filename)
                    
                    # Convert to WAV bytes
                    wav_io = io.BytesIO()
                    audio.export(wav_io, format="wav")
                    wav_data = wav_io.getvalue()
                    
                    return wav_data
                    
                except Exception as pydub_error:
                    logger.error(f"pydub conversion failed: {pydub_error}")
                    logger.info("Falling back to direct MP3 to WAV conversion...")
                    
                    # Fallback: try to convert MP3 to WAV manually using basic method
                    return self._mp3_to_wav_fallback(tmp_mp3_filename)
                    
            finally:
                if os.path.exists(tmp_mp3_filename):
                    os.unlink(tmp_mp3_filename)
                    
        except Exception as e:
            logger.error(f"Google TTS failed: {e}")
            raise RuntimeError(f"Google TTS conversion failed: {e}")

    def _mp3_to_wav_fallback(self, mp3_filename: str) -> bytes:
        """Fallback MP3 to WAV conversion using subprocess"""
        import subprocess
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            wav_filename = tmp_wav.name
            
        try:
            # Try using ffmpeg directly via subprocess
            result = subprocess.run([
                'ffmpeg', '-i', mp3_filename, '-acodec', 'pcm_s16le', 
                '-ar', str(self.sample_rate), '-ac', '1', wav_filename, '-y'
            ], capture_output=True, text=True, check=True)
            
            with open(wav_filename, 'rb') as f:
                wav_data = f.read()
            
            logger.info("Successfully converted MP3 to WAV using ffmpeg subprocess")
            return wav_data
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"ffmpeg subprocess conversion failed: {e}")
            raise RuntimeError(
                f"Both pydub and ffmpeg conversion failed. "
                f"Please ensure ffmpeg is installed: sudo apt install ffmpeg"
            )
        finally:
            if os.path.exists(wav_filename):
                os.unlink(wav_filename)

    def _pyttsx3_to_wav(self, text: str) -> bytes:
        """Convert text to WAV using pyttsx3"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_filename = tmp_file.name

        try:
            self.engine.save_to_file(text, tmp_filename)
            self.engine.runAndWait()

            with open(tmp_filename, 'rb') as f:
                wav_data = f.read()
            return wav_data

        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)

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

    def resample_audio(self, pcm_data: bytes, original_sr: int, target_sr: int, channels: int) -> bytes:
        """Resample audio to target sample rate"""
        if original_sr == target_sr:
            return pcm_data

        audio_array = np.frombuffer(pcm_data, dtype=np.int16)

        if channels == 2:
            audio_array = audio_array.reshape(-1, 2)

        ratio = target_sr / original_sr
        new_length = int(len(audio_array) * ratio)

        if channels == 1:
            resampled = np.interp(
                np.linspace(0, len(audio_array) - 1, new_length),
                np.arange(len(audio_array)),
                audio_array
            ).astype(np.int16)
        else:
            resampled = np.zeros((new_length, 2), dtype=np.int16)
            for ch in range(2):
                resampled[:, ch] = np.interp(
                    np.linspace(0, len(audio_array) - 1, new_length),
                    np.arange(len(audio_array)),
                    audio_array[:, ch]
                ).astype(np.int16)
            resampled = resampled.flatten()

        return resampled.tobytes()

    def text_to_frame_pcm_s16le(self, text: str, pts: int = 0, force_mono: bool = True) -> FramePcmS16le:
        """
        Convert text to FramePcmS16le format - optimized for streaming

        Args:
            text: Text to convert to speech
            pts: Presentation timestamp
            force_mono: Force conversion to mono audio

        Returns:
            FramePcmS16le protobuf message
        """
        logger.debug(f"Converting text to speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Try direct PCM conversion first (avoids WAV processing)
        try:
            pcm_data, original_sr, channels = self.text_to_pcm_direct(text)
            logger.debug(f"Direct PCM: {original_sr}Hz, {channels}ch, {len(pcm_data)} bytes")
            
        except Exception as e:
            logger.warning(f"Direct PCM failed, falling back to WAV method: {e}")
            # Fallback to WAV method
            wav_data = self.text_to_wav_bytes(text)
            pcm_data, original_sr, channels = self.wav_to_pcm_s16le(wav_data)
            logger.debug(f"WAV fallback: {original_sr}Hz, {channels}ch, {len(pcm_data)} bytes")

        # Handle stereo to mono conversion if needed
        if force_mono and channels == 2:
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            stereo_array = audio_array.reshape(-1, 2)
            mono_array = np.mean(stereo_array, axis=1).astype(np.int16)
            pcm_data = mono_array.tobytes()
            channels = 1

        # Resample only if necessary
        if original_sr != self.sample_rate:
            pcm_data = self.resample_audio(pcm_data, original_sr, self.sample_rate, channels)

        # Create the FramePcmS16le protobuf message
        frame = FramePcmS16le(
            data=pcm_data,
            pts=pts,
            time_base=FramePcmS16le.Fraction(num=1, den=self.sample_rate),
            channels=channels
        )
        
        duration_seconds = len(pcm_data) / (2 * channels * self.sample_rate)
        logger.debug(f"Generated frame: {self.sample_rate}Hz, {channels}ch, {len(pcm_data)} bytes, {duration_seconds:.2f}s")

        return frame

    def text_to_wav_bytes(self, text: str) -> bytes:
        """
        Convert text to WAV format bytes (fallback method)

        Args:
            text: Text to convert to speech

        Returns:
            WAV audio data as bytes
        """
        if self.use_gtts:
            return self._gtts_to_wav(text)
        else:
            return self._pyttsx3_to_wav(text)


def main():
    logger.info("Starting TTS service...")
    
    # Example: List available voices (uncomment to see options)
    # TextToFramePcm.list_gtts_voices()
    # TextToFramePcm.list_pyttsx3_voices()
    
    message_encoder = ProtobufEncoder(message_type=FramePcmS16le)
    zenoh_interface = ZenohInterface(name="zenoh")

    publisher = zenoh_interface.get_publisher("tts_audio")
    subscriber = zenoh_interface.get_subscriber("tts_text")

    # Initialize converter with voice selection
    try:
        converter = TextToFramePcm(
            sample_rate=16000,
            prefer_gtts=True,
            language='en',
            # For lower pitch with gTTS, try these options:
            tld='co.uk',  # British English (lower pitch than 'com')
            # tld='com.au',  # Australian English (moderate pitch)
            # tld='ca',      # Canadian English (moderate pitch)
            
            # For pyttsx3 fallback (uncomment to specify voice):
            # voice_id='HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0'
            slow=True
        )
        logger.info("TTS service initialized successfully")
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

            logger.info(f"Published TTS audio ({conversion_time:.2f}s conversion time)")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            continue


if __name__ == "__main__":
    # Uncomment these lines to see available voices before starting:
    # print("=== Available Voices ===")
    # TextToFramePcm.list_gtts_voices()
    # print()
    # TextToFramePcm.list_pyttsx3_voices()
    # print("========================")
    
    main()