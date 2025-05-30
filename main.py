import asyncio
import json
import logging
from datetime import datetime
from audio_input.mic_or_file_streamer import stream_from_mic, stream_from_file
from transcribe_stream.transcribe_client import start_transcription
from transcript_store.rolling_transcript import RollingTranscript
from summarizer.summarizer import generate_minute_summary, generate_summary, generate_title
from course_generator import generate_course_suggestions

# Configure logging
logger = logging.getLogger("MainPipeline")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

async def main(use_mic: bool = True, audio_file: str = None, summarize_interval: int = 60):
    transcript_store = RollingTranscript(window_seconds=300)
    transcript_file = "transcript.json"
    output_file = "output/summary.json"
    course_output_file = "output/course_suggestions.json"
    minute_summaries = []
    output_data = {"minute_summaries": [], "overall": {}}
    last_processed_minute = -1

    async def transcript_callback(text: str, timestamp: float):
        logger.info(f"[{datetime.utcnow().isoformat()}] Processing transcript at timestamp {timestamp}")
        transcript_store.add_transcript(text, timestamp)
        current_minute = int(timestamp // 60)

        if not hasattr(transcript_callback, 'last_summarized_minute'):
            transcript_callback.last_summarized_minute = current_minute - 1

        if current_minute > transcript_callback.last_summarized_minute:
            prev_minute = transcript_callback.last_summarized_minute
            minute_transcript = transcript_store.get_minute_segment(prev_minute)
            if minute_transcript:
                logger.info(f"[{datetime.utcnow().isoformat()}] Generating summary for minute {prev_minute}")
                minute_summary = generate_minute_summary(minute_transcript, minute_summaries)
                logger.info(f"[{datetime.utcnow().isoformat()}] Minute {prev_minute} Summary: {minute_summary}")
                minute_summaries.append(minute_summary)
                output_data["minute_summaries"].append({
                    "minute": prev_minute,
                    "summary": minute_summary
                })
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(output_data, f, indent=2)
                    logger.info(f"[{datetime.utcnow().isoformat()}] Saved summary to {output_file}")
                except Exception as e:
                    logger.error(f"[{datetime.utcnow().isoformat()}] Error saving minute summary: {e}")
            transcript_callback.last_summarized_minute = current_minute

    async def monitor_summaries():
        nonlocal last_processed_minute
        while True:
            logger.info(f"[{datetime.utcnow().isoformat()}] Checking for new summaries")
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                minute_summaries = data.get("minute_summaries", [])
                if minute_summaries:
                    current_latest_minute = max(entry["minute"] for entry in minute_summaries)
                    if current_latest_minute > last_processed_minute:
                        logger.info(f"[{datetime.utcnow().isoformat()}] New minute summary detected: {current_latest_minute}")
                        last_processed_minute = current_latest_minute
                        try:
                            course_suggestions = await generate_course_suggestions(json_path=output_file)
                            logger.info(f"[{datetime.utcnow().isoformat()}] Course Suggestions for minute {current_latest_minute}: {json.dumps(course_suggestions, indent=2)}")
                            try:
                                existing_suggestions = []
                                try:
                                    with open(course_output_file, 'r', encoding='utf-8') as f:
                                        existing_suggestions = json.load(f)
                                    if not isinstance(existing_suggestions, list):
                                        existing_suggestions = []
                                except FileNotFoundError:
                                    pass
                                existing_suggestions.append(course_suggestions)
                                with open(course_output_file, 'w', encoding='utf-8') as f:
                                    json.dump(existing_suggestions, f, indent=2)
                                logger.info(f"[{datetime.utcnow().isoformat()}] Course suggestions appended to {course_output_file}")
                            except Exception as e:
                                logger.error(f"[{datetime.utcnow().isoformat()}] Error saving course suggestions: {e}")
                        except Exception as e:
                            logger.error(f"[{datetime.utcnow().isoformat()}] Error generating course suggestions: {e}")
            except Exception as e:
                logger.error(f"[{datetime.utcnow().isoformat()}] Error monitoring summaries: {e}")
            await asyncio.sleep(10)

    # Choose audio source
    audio_stream = stream_from_mic() if use_mic else stream_from_file(audio_file)

    # Start monitoring summaries task BEFORE transcription
    logger.info(f"[{datetime.utcnow().isoformat()}] Starting summary monitoring task")
    monitor_task = asyncio.create_task(monitor_summaries())

    # Run transcription
    logger.info(f"[{datetime.utcnow().isoformat()}] Starting transcription")
    try:
        final_transcript = await start_transcription(audio_stream, transcript_callback)
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            logger.info("Monitoring task cancelled.")

    # Save final transcripts
    transcript_store.save_to_file(transcript_file)

    # Generate final minute summary
    last_minute = max(transcript_store.get_all_minute_keys(), default=0)
    last_transcript = transcript_store.get_minute_segment(last_minute)
    if last_transcript:
        logger.info(f"[{datetime.utcnow().isoformat()}] Generating final summary for minute {last_minute}")
        minute_summary = generate_minute_summary(last_transcript, minute_summaries)
        minute_summaries.append(minute_summary)
        output_data["minute_summaries"].append({
            "minute": last_minute,
            "summary": minute_summary
        })

    # Generate overall summary and title
    full_transcript = " ".join([
        transcript_store.get_minute_segment(minute_key)
        for minute_key in transcript_store.get_all_minute_keys()
    ])
    if full_transcript:
        logger.info(f"[{datetime.utcnow().isoformat()}] Generating overall summary")
        overall_summary = generate_summary(full_transcript, [])
        overall_title = generate_title(full_transcript, overall_summary)
        output_data["overall"] = {
            "title": overall_title,
            "summary": overall_summary
        }

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"[{datetime.utcnow().isoformat()}] Final summary output saved to {output_file}")
    except Exception as e:
        logger.error(f"[{datetime.utcnow().isoformat()}] Error saving final summary output: {e}")

    # Run final course suggestions
    try:
        course_suggestions = await generate_course_suggestions(json_path=output_file)
        logger.info(f"[{datetime.utcnow().isoformat()}] Final Course Suggestions: {json.dumps(course_suggestions, indent=2)}")
        try:
            with open(course_output_file, 'w', encoding='utf-8') as f:
                json.dump([course_suggestions], f, indent=2)
            logger.info(f"[{datetime.utcnow().isoformat()}] Final course suggestions saved to {course_output_file}")
        except Exception as e:
            logger.error(f"[{datetime.utcnow().isoformat()}] Error saving final course suggestions: {e}")
    except Exception as e:
        logger.error(f"[{datetime.utcnow().isoformat()}] Error running final course suggestion: {e}")

if __name__ == "__main__":
    asyncio.run(main(use_mic=True, audio_file="sample_audio.wav"))
