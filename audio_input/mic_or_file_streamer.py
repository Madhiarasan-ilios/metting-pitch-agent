import asyncio
import sounddevice as sd
import numpy as np
import wave
import logging
from typing import AsyncGenerator

logging.basicConfig(level=logging.INFO)

async def stream_from_mic(sample_rate: int = 16000, chunk_size: int = 1024) -> AsyncGenerator[bytes, None]:
    """Stream audio from microphone in chunks."""
    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    def audio_callback(indata: np.ndarray, frames: int, time, status):
        loop.call_soon_threadsafe(input_queue.put_nowait, indata.tobytes())

    stream = sd.InputStream(
        channels=1,
        samplerate=sample_rate,
        dtype='int16',
        blocksize=chunk_size,
        callback=audio_callback
    )

    with stream:
        while True:
            indata = await input_queue.get()
            yield indata

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