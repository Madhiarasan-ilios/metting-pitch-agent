MINUTE_SUMMARY_PROMPT = """
You are a meeting assistant. Summarize the following 1-minute transcript segment in a concise paragraph (50-100 words). Include key topics discussed. If provided, use the previous summaries to maintain context and continuity, but focus on the current segment:
Current 1-minute transcript: "{current_transcript}"
Previous summaries (if any): "{previous_summaries}"
"""

SUMMARY_PROMPT = """
You are a meeting assistant. Summarize the following full transcript in a concise paragraph (100-150 words). Use bullet points for key points if appropriate. Be specific, professional, and focus on the main topics discussed:
"{transcript}"
"""

TITLE_PROMPT = """
Generate a short, relevant title (5-10 words) for this meeting based on the transcript and its summary. Ensure the title is concise and captures the main focus:
Transcript: "{transcript}"
Summary: "{summary}"
"""