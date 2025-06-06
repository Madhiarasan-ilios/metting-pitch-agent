import asyncio
import sounddevice as sd
import numpy as np
import wave
import logging
from typing import AsyncGenerator

logging.basicConfig(level=logging.INFO)

async def stream_from_mic(sample_rate: int = 16000, chunk_size: int = 1024, input_type: str = 'mic', device: int = None) -> AsyncGenerator[bytes, None]:
    """
    Stream audio from either microphone or speaker (loopback) in chunks.
    
    Args:
        sample_rate: Sampling rate in Hz (default: 16000).
        chunk_size: Number of frames per chunk (default: 1024).
        input_type: 'mic' for microphone or 'speaker' for loopback audio (default: 'mic').
        device: Specific device index to use (default: None, uses default device).
    """
    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    def audio_callback(indata: np.ndarray, frames: int, time, status):
        if status:
            logging.warning(f"Stream status: {status}")
        loop.call_soon_threadsafe(input_queue.put_nowait, indata.tobytes())

    # Configure stream based on input_type
    try:
        if input_type == 'mic':
            stream = sd.InputStream(
                channels=1,
                samplerate=sample_rate,
                dtype='int16',
                blocksize=chunk_size,
                device=device,
                callback=audio_callback
            )
        elif input_type == 'speaker':
            # Use loopback for speaker capture
            stream = sd.InputStream(
                channels=1,
                samplerate=sample_rate,
                dtype='int16',
                blocksize=chunk_size,
                device=device,
                callback=audio_callback,
                extra_settings=sd.WasapiSettings(exclusive=False, loopback=True) if sd.get_portaudio_version()[1].startswith('WASAPI') else None
            )
        else:
            logging.error("Invalid input_type. Use 'mic' or 'speaker'.")
            return

        with stream:
            while True:
                indata = await input_queue.get()
                yield indata

    except Exception as e:
        logging.error(f"Error in audio stream: {e}")
        return

async def stream_from_file(file_path: str, chunk_size: int = 1024) -> AsyncGenerator[bytes, None]:
    """Stream audio from a WAV file in chunks."""
    try:
        with wave.open(file_path, 'rb') as wf:
            if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                logging.error("Audio file must be 16kHz, mono, 16-bit PCM")
                return
            while True:
                data = wf.readframes(chunk_size)
                if not data:
                    break
                yield data
                await asyncio.sleep(0.01)  # Simulate real-time streaming
    except FileNotFoundError:
        logging.error(f"Audio file not found: {file_path}")