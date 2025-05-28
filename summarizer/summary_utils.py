import re
import yake
import logging

logging.basicConfig(level=logging.INFO)

def clean_text(text: str) -> str:
    """Clean transcript text by removing extra spaces and special characters."""
    text = re.sub(r'\s+', ' ', text.strip())  # Normalize whitespace
    text = re.sub(r'[^\w\s.,!?]', '', text)   # Remove special characters
    return text

def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Extract keywords from text using YAKE."""
    try:
        kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=max_keywords)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]
    except Exception as e:
        logging.error(f"Keyword extraction error: {e}")
        return []