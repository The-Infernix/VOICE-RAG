import sys
from pathlib import Path
import pytest
import numpy as np

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from api.schemas import (
    AskRequest, AskResponse, Answer, Chunk, RetrievalResult,
    GuardResult, StageTrace, LatencyBreakdown, Citation,
    QueryType, Language, VoiceRequest, HealthResponse,
)


@pytest.fixture
def sample_chunk():
    return Chunk(
        chunk_id="chunk_001",
        document_id="doc_001",
        text="The capital of India is New Delhi. It is located on the banks of the Yamuna river.",
        metadata={"language": "en", "strategy": "passage_native"},
        score=0.95,
    )


@pytest.fixture
def sample_chunks():
    return [
        Chunk(chunk_id="c1", document_id="d1", text="Delhi is the capital of India.", metadata={"language": "en"}, score=0.95),
        Chunk(chunk_id="c2", document_id="d2", text="Mumbai is the financial capital.", metadata={"language": "en"}, score=0.88),
        Chunk(chunk_id="c3", document_id="d3", text="Bangalore is a tech hub.", metadata={"language": "en"}, score=0.72),
    ]


@pytest.fixture
def sample_answer():
    return Answer(
        text="Delhi is the capital of India.",
        method="extractive",
        citations=[Citation(chunk_id="c1", text="Delhi is the capital.", score=0.95)],
        confidence=0.95,
    )


@pytest.fixture
def hindi_chunks():
    return [
        Chunk(chunk_id="h1", document_id="d1", text="भारत की राजधानी नई दिल्ली है।", metadata={"language": "hi"}, score=0.93),
        Chunk(chunk_id="h2", document_id="d2", text="मुंबई भारत का वित्तीय केंद्र है।", metadata={"language": "hi"}, score=0.85),
    ]


@pytest.fixture
def gujarati_chunks():
    return [
        Chunk(chunk_id="g1", document_id="d1", text="ભારતની રાજધાની નવી દિલ્હી છે.", metadata={"language": "gu"}, score=0.91),
        Chunk(chunk_id="g2", document_id="d2", text="મુંબઈ ભારતનું નાણાકીય કેન્દ્ર છે.", metadata={"language": "gu"}, score=0.82),
    ]


@pytest.fixture
def short_text():
    return "The quick brown fox jumps over the lazy dog."


@pytest.fixture
def long_text():
    return (
        "India, officially the Republic of India, is a country in South Asia. "
        "It is the world's most populous country and the seventh-largest by area. "
        "The country is bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, "
        "and the Bay of Bengal on the southeast. "
        "India shares land borders with Pakistan to the west, China, Nepal, and Bhutan to the northeast, "
        "and Bangladesh and Myanmar to the east. "
        "Sri Lanka and the Maldives are located to its south. "
        "India's capital is New Delhi, while Mumbai is its largest city. "
        "Other major cities include Bangalore, Hyderabad, Chennai, and Kolkata. "
        "The country has a diverse geography, ranging from the Himalayan mountain range in the north "
        "to the tropical beaches of Goa in the west. "
        "India is a federal republic with 28 states and 8 union territories. "
        "Hindi and English are the official languages of the central government. "
        "The Indian rupee is the official currency. "
        "India's economy is the fifth-largest in the world by nominal GDP. "
        "The country is a major exporter of IT services, pharmaceuticals, and textiles. "
        "India's space program is one of the most advanced in the world. "
        "The Indian Space Research Organisation (ISRO) has successfully launched missions to the Moon and Mars."
    )


@pytest.fixture
def empty_text():
    return ""


@pytest.fixture
def whitespace_text():
    return "   \n  \t  "
