import time
from collections import deque
import json
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)

class RollingTranscript:
    """Store a rolling window of transcripts and organize by 1-minute segments."""
    def __init__(self, window_seconds: int = 300):  # 5 minutes for full transcript
        self.transcripts = deque(maxlen=1000)  # Store individual entries
        self.window_seconds = window_seconds
        self.minute_segments = {}  # Store transcripts by minute

    def add_transcript(self, text: str, timestamp: float):
        """Add a transcript with timestamp and organize into minute segments."""
        self.transcripts.append({"timestamp": timestamp, "text": text})
        # Group by minute (floor of timestamp / 60)
        minute_key = int(timestamp // 60)
        if minute_key not in self.minute_segments:
            self.minute_segments[minute_key] = []
        self.minute_segments[minute_key].append(text)

    def get_transcripts(self) -> List[Dict]:
        """Get transcripts within the time window."""
        current_time = time.time()
        return [
            t for t in self.transcripts
            if current_time - t["timestamp"] <= self.window_seconds
        ]

    def get_minute_segment(self, minute_key: int) -> str:
        """Get concatenated transcript text for a specific minute."""
        return " ".join(self.minute_segments.get(minute_key, []))

    def get_all_minute_keys(self) -> List[int]:
        """Get all minute keys in ascending order."""
        return sorted(self.minute_segments.keys())

    def save_to_file(self, file_path: str):
        """Save current transcripts to a JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(list(self.transcripts), f, indent=2)
        except Exception as e:
            logging.error(f"Error saving transcripts: {e}")