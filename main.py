import asyncio
import json
import logging
import time
from audio_input.mic_or_file_streamer import stream_from_mic, stream_from_file
from transcribe_stream.transcribe_client import start_transcription
from transcript_store.rolling_transcript import RollingTranscript
from summarizer.summarizer import generate_minute_summary, generate_summary, generate_title
from summarizer.summary_utils import extract_keywords

logging.basicConfig(level=logging.INFO)

async def main(use_mic: bool = True, audio_file: str = None, summarize_interval: int = 60):
    """Orchestrate audio streaming, transcription, and real-time summarization."""
    transcript_store = RollingTranscript(window_seconds=300)
    transcript_file = "transcript.json"
    output_file = "output/summary.json"
    minute_summaries = []  
    output_data = {"minute_summaries": [], "overall": {}}

    async def transcript_callback(text: str, timestamp: float):
        """Handle incoming transcripts and trigger per-minute summaries."""
        transcript_store.add_transcript(text, timestamp)
        current_minute = int(timestamp // 60)
        
        # Check if a new minute has started
        if not hasattr(transcript_callback, 'last_summarized_minute'):
            transcript_callback.last_summarized_minute = current_minute - 1

        if current_minute > transcript_callback.last_summarized_minute:
            # Summarize the previous minute
            prev_minute = transcript_callback.last_summarized_minute
            minute_transcript = transcript_store.get_minute_segment(prev_minute)
            if minute_transcript:
                # Generate summary with previous summaries as context
                minute_summary = generate_minute_summary(minute_transcript, minute_summaries)
                minute_keywords = extract_keywords(minute_transcript)
                logging.info(f"Minute {prev_minute} Summary: {minute_summary}")
                logging.info(f"Minute {prev_minute} Topics: {minute_keywords}")
                minute_summaries.append(minute_summary)
                output_data["minute_summaries"].append({
                    "minute": prev_minute,
                    "summary": minute_summary,
                    "topics": minute_keywords
                })
                # Save intermediate output
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2)
                except Exception as e:
                    logging.error(f"Error saving minute summary: {e}")
            transcript_callback.last_summarized_minute = current_minute

    # Choose audio source
    audio_stream = stream_from_mic() if use_mic else stream_from_file(audio_file)

    # Start transcription
    final_transcript = await start_transcription(audio_stream, transcript_callback)

    # Save final transcripts
    transcript_store.save_to_file(transcript_file)

    # Generate final minute summary if needed
    last_minute = max(transcript_store.get_all_minute_keys(), default=0)
    last_transcript = transcript_store.get_minute_segment(last_minute)
    if last_transcript:
        minute_summary = generate_minute_summary(last_transcript, minute_summaries)
        minute_keywords = extract_keywords(last_transcript)
        logging.info(f"Minute {last_minute} Summary: {minute_summary}")
        logging.info(f"Minute {last_minute} Topics: {minute_keywords}")
        minute_summaries.append(minute_summary)
        output_data["minute_summaries"].append({
            "minute": last_minute,
            "summary": minute_summary,
            "topics": minute_keywords
        })

    # Generate overall summary and title
    full_transcript = transcript_store.get_minute_segment(0)
    for minute_key in transcript_store.get_all_minute_keys()[1:]:
        full_transcript += " " + transcript_store.get_minute_segment(minute_key)
    if full_transcript:
        overall_keywords = extract_keywords(full_transcript)
        overall_summary = generate_summary(full_transcript, overall_keywords)
        overall_title = generate_title(full_transcript, overall_summary)
        logging.info(f"Overall Summary: {overall_summary}")
        logging.info(f"Overall Title: {overall_title}")
        output_data["overall"] = {
            "title": overall_title,
            "summary": overall_summary,
            "keywords": overall_keywords
        }

    # Save final output
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        logging.info(f"Final output saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving final output: {e}")

if __name__ == "__main__":
    # Set to False and provide a file path to use a pre-recorded file
    asyncio.run(main(use_mic=True, audio_file="sample_audio.wav"))