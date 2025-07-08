import wikipediaapi
import re
import random
import streamlit as st
from config import get_genai_client, GENAI_MODEL, PROMPT_VARIANTS


def fetch_wikipedia_content(topic):
    """Fetch Wikipedia content for a given topic"""
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='YourAppName/1.0 (your_email@example.com)', language="en")
    try:
        page = wiki_wiki.page(topic)
        if not page.exists():
            return None
        return page.text[:5000]  # Limit text size to avoid overloading
    except Exception as e:
        st.error(f"Wikipedia fetch failed: {e}")
        return None


def generate_questions_from_text(text, num_questions, variant=False):
    """Generate questions using Google's Gemini API"""
    if variant:
        template = random.choice(PROMPT_VARIANTS)
        prompt = template.format(n=num_questions, text=text[:1500])
    else:
        prompt = f"Based on the following content, generate exactly {num_questions} clear and concise questions. Only academic & research based questions.\n\n{text[:1500]}"

    try:
        genai_client = get_genai_client()
        resp = genai_client.models.generate_content(
            model=GENAI_MODEL,
            contents=prompt
        )
        raw_output = resp.text
    except Exception as e:
        st.error(f"LLM call failed: {e}")
        return []

    # Parse numbered or bullet lines into question strings
    possible = re.findall(
        r'(?:\d+\.\s*|[-*]\s*|^)(.*?)(?=\n|$)',
        raw_output.strip(),
        re.MULTILINE
    )
    questions = [
        q.strip() for q in possible
        if q.strip() and (
                '?' in q or
                q.lower().startswith(('what', 'how', 'why'))
        )
    ]
    return questions[:num_questions]