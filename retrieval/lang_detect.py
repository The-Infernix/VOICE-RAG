import re

_GUJARATI_RE = re.compile(r"[\u0A80-\u0AFF]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def detect_query_language(text: str) -> str:
    if not text:
        return "en"
    if _GUJARATI_RE.search(text):
        return "gu"
    if _DEVANAGARI_RE.search(text):
        return "hi"
    return "en"
