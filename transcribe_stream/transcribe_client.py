import asyncio
import logging
import time
from typing import AsyncGenerator
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent

logging.basicConfig(level=logging.INFO)

class TranscriptionHandler(TranscriptResultStreamHandler):
    """Handle AWS Transcribe streaming events."""
    def __init__(self, output_stream, transcript_callback):
        super().__init__(output_stream)
        self.transcript_callback = transcript_callback
        self.final_transcript = ""

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        for result in results:
            if result.is_partial:
                continue
            for alt in result.alternatives:
                self.final_transcript += alt.transcript + " "
                # Pass timestamp and transcript to callback
                await self.transcript_callback(alt.transcript, time.time())

async def start_transcription(audio_stream: AsyncGenerator[bytes, None], transcript_callback, region: str = "ap-south-1"):
    """Start AWS Transcribe streaming session."""
    try:
        client = TranscribeStreamingClient(region=region)
        stream = await client.start_stream_transcription(
            language_code="en-US",
            show_speaker_label=True,
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        handler = TranscriptionHandler(stream.output_stream, transcript_callback)

        async def write_chunks():
            async for chunk in audio_stream:
                await stream.input_stream.send_audio_event(audio_chunk=chunk)
            await stream.input_stream.end_stream()

        await asyncio.gather(
            write_chunks(),
            handler.handle_events()
        )
        return handler.final_transcript.strip()

    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return ""