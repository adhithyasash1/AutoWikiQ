import os
from google import genai

# Configuration constants
PROMPT_VARIANTS = [
    "Based on the content below, generate exactly {n} clear and concise questions. Focus on academic & research angles:\n\n{text}",
    "Please produce {n} precise and scholarly questions from the following text:\n\n{text}",
    "From the excerpt below, craft exactly {n} rigorous academic questions:\n\n{text}"
]

# API Configuration
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
GENAI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Initialize Gemini client
def get_genai_client():
    if not GENAI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set! Run `export GEMINI_API_KEY=...` and restart.")
    return genai.Client(api_key=GENAI_API_KEY)

# File names
FEEDBACK_FILE = "questions_feedback.csv"
HISTORY_FILE = "acceptance_history.csv"
CLASSIFIED_FILE = "classified_questions.csv"
MODEL_FILE = "question_classifier.pkl"