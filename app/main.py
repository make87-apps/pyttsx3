import logging
import time
from make87_messages.audio.frame_pcm_s16le_pb2 import FramePcmS16le
from make87.encodings import ProtobufEncoder
from make87.interfaces.zenoh import ZenohInterface

import pyttsx3
import io
import wave
import numpy as np

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    format="[%(asctime)sZ %(levelname)s  %(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%dT%H:%M:%S"
)


def main():
    message_encoder = ProtobufEncoder(message_type=FramePcmS16le)
    zenoh_interface = ZenohInterface(name="zenoh")

    publisher = zenoh_interface.get_publisher("tts_audio")
    subscriber = zenoh_interface.get_subscriber("tts_text")

    # Initialize converter with 16kHz sample rate (common for speech)
    converter = TextToFramePcm(sample_rate=16000)

    for msg in subscriber:
        text = msg.payload.to_bytes().decode("utf-8")
        frame = converter.text_to_frame_pcm_s16le(text, pts=0)
        message_encoded = message_encoder.encode(frame)
        publisher.put(payload=message_encoded)

        logging.info(f"Published: {text}")
        time.sleep(1)


if __name__ == "__main__":
    main()


class TextToFramePcm:
    """Converts text to FramePcmS16le format using pyttsx3"""

    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the text-to-speech converter

        Args:
            sample_rate: Audio sample rate (default: 16000 Hz)
        """
        self.sample_rate = sample_rate
        self.engine = pyttsx3.init()

        # Configure TTS engine for better quality
        self.engine.setProperty('rate', 150)  # Speech rate
        self.engine.setProperty('volume', 1.0)  # Volume level

        # Try to set a higher quality voice if available
        voices = self.engine.getProperty('voices')
        if voices:
            # Use the first available voice, or specify a preferred one
            self.engine.setProperty('voice', voices[0].id)

    def text_to_wav_bytes(self, text: str) -> bytes:
        """
        Convert text to WAV format bytes using pyttsx3

        Args:
            text: Text to convert to speech

        Returns:
            WAV audio data as bytes
        """
        # Create a temporary file-like object in memory
        temp_file = io.BytesIO()

        # Save speech to temporary file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_filename = tmp_file.name

        try:
            # Generate speech and save to temporary file
            self.engine.save_to_file(text, tmp_filename)
            self.engine.runAndWait()

            # Read the WAV file
            with open(tmp_filename, 'rb') as f:
                wav_data = f.read()

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)

        return wav_data

    def wav_to_pcm_s16le(self, wav_data: bytes) -> tuple[bytes, int, int]:
        """
        Convert WAV data to PCM S16LE format

        Args:
            wav_data: WAV format audio data

        Returns:
            Tuple of (pcm_data, sample_rate, channels)
        """
        # Parse WAV file from bytes
        wav_io = io.BytesIO(wav_data)

        with wave.open(wav_io, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()

            # Convert to numpy array for processing
            if sample_width == 1:
                # 8-bit audio
                audio_data = np.frombuffer(frames, dtype=np.uint8)
                # Convert to signed 16-bit
                audio_data = ((audio_data.astype(np.float32) - 128) * 256).astype(np.int16)
            elif sample_width == 2:
                # 16-bit audio
                audio_data = np.frombuffer(frames, dtype=np.int16)
            elif sample_width == 4:
                # 32-bit audio - convert to 16-bit
                audio_data = np.frombuffer(frames, dtype=np.int32)
                audio_data = (audio_data // 65536).astype(np.int16)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            # Ensure little-endian format
            pcm_data = audio_data.astype('<i2').tobytes()  # '<i2' means little-endian 16-bit signed int

            return pcm_data, sample_rate, channels

    def resample_audio(self, pcm_data: bytes, original_sr: int, target_sr: int, channels: int) -> bytes:
        """
        Resample audio to target sample rate using simple linear interpolation

        Args:
            pcm_data: PCM audio data
            original_sr: Original sample rate
            target_sr: Target sample rate
            channels: Number of audio channels

        Returns:
            Resampled PCM data
        """
        if original_sr == target_sr:
            return pcm_data

        # Convert bytes to numpy array
        audio_array = np.frombuffer(pcm_data, dtype=np.int16)

        if channels == 2:
            # Stereo - reshape to handle channels
            audio_array = audio_array.reshape(-1, 2)

        # Calculate resampling ratio
        ratio = target_sr / original_sr
        new_length = int(len(audio_array) * ratio)

        if channels == 1:
            # Mono resampling
            resampled = np.interp(
                np.linspace(0, len(audio_array) - 1, new_length),
                np.arange(len(audio_array)),
                audio_array
            ).astype(np.int16)
        else:
            # Stereo resampling
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
        Convert text to FramePcmS16le format

        Args:
            text: Text to convert to speech
            pts: Presentation timestamp
            force_mono: Force conversion to mono audio

        Returns:
            FramePcmS16le object containing the audio data
        """
        # Convert text to WAV
        wav_data = self.text_to_wav_bytes(text)

        # Extract PCM data
        pcm_data, original_sr, channels = self.wav_to_pcm_s16le(wav_data)

        # Convert stereo to mono if requested
        if force_mono and channels == 2:
            # Convert stereo to mono by averaging channels
            audio_array = np.frombuffer(pcm_data, dtype=np.int16)
            stereo_array = audio_array.reshape(-1, 2)
            mono_array = np.mean(stereo_array, axis=1).astype(np.int16)
            pcm_data = mono_array.tobytes()
            channels = 1

        # Resample if necessary
        if original_sr != self.sample_rate:
            pcm_data = self.resample_audio(pcm_data, original_sr, self.sample_rate, channels)

        # Create the FramePcmS16le structure
        frame = FramePcmS16le(
            data=pcm_data,
            pts=pts,
            time_base=FramePcmS16le.Fraction(num=1, den=self.sample_rate),
            channels=channels
        )

        return frame
