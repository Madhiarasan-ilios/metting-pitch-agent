import asyncio
import logging
from audio_input.mic_or_file_streamer import stream_from_mic, stream_from_file
from transcribe_stream.transcribe_client import start_transcription
from transcript_store.rolling_transcript import RollingTranscript

logging.basicConfig(level=logging.INFO)

async def main(use_mic: bool = True, audio_file: str = None):
    """Main function to orchestrate audio streaming and transcription."""
    transcript_store = RollingTranscript(window_seconds=300)

    async def transcript_callback(text: str):
        """Callback to handle incoming transcripts."""
        logging.info(f"Transcript: {text}")
        transcript_store.add_transcript(text)
        transcript_store.save_to_file("transcripts.json")

    # Choose audio source
    audio_stream = stream_from_mic() if use_mic else stream_from_file(audio_file)

    # Start transcription
    final_transcript = await start_transcription(audio_stream, transcript_callback)

    # Log final transcript
    logging.info(f"Final Transcript: {final_transcript}")

    # Save final transcripts
    transcript_store.save_to_file("transcripts.json")

if __name__ == "__main__":
    asyncio.run(main(use_mic=True, audio_file="sample_audio.wav"))