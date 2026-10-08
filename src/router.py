import re

GREETINGS = [
    r"^hi\b", r"^hello\b", r"^hey\b", r"^good (morning|afternoon|evening)",
    r"^who are you", r"^what can you do", r"^help\b", r"^thanks?", r"^thank you"
]

def check_intent(query: str):
    """
    Checks if a query is simple conversational small talk.
    Returns (is_casual: bool, canned_response: str).
    This saves 100% of retrieval and context tokens on casual queries.
    """
    clean_q = query.strip().lower()
    
    # check greeting patterns
    for pat in GREETINGS:
        if re.search(pat, clean_q):
            if "who are you" in clean_q or "what can you do" in clean_q or "help" in clean_q:
                return True, (
                    "Hello! I am DocuQuery, your technical documentation assistant. "
                    "I can answer questions and provide exact code syntax for FastAPI, Docker, "
                    "Git, and PostgreSQL documentation with source citations."
                )
            if "thank" in clean_q:
                return True, "You're welcome! Let me know if you need any other documentation lookups."
            return True, "Hello! How can I help you with FastAPI, Docker, Git, or PostgreSQL today?"

    return False, ""
