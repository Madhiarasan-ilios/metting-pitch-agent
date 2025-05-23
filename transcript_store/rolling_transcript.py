import time
from collections import deque
import json
import logging

logging.basicConfig(level=logging.INFO)

class RollingTranscript:
    """Store a rolling window of transcripts."""
    def __init__(self, window_seconds: int = 300):  # 5 minutes
        self.transcripts = deque(maxlen=1000)  # Arbitrary large limit
        self.window_seconds = window_seconds

    def add_transcript(self, text: str):
        """Add a transcript with timestamp."""
        timestamp = time.time()
        self.transcripts.append({"timestamp": timestamp, "text": text})

    def get_transcripts(self) -> list:
        """Get transcripts within the time window."""
        current_time = time.time()
        return [
            t for t in self.transcripts
            if current_time - t["timestamp"] <= self.window_seconds
        ]

    def save_to_file(self, file_path: str):
        """Save current transcripts to a JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(list(self.transcripts), f, indent=2)
        except Exception as e:
            logging.error(f"Error saving transcripts: {e}")