import pytest
from api.schemas import Chunk, Answer, Citation
from generation.extractive import ExtractiveGenerator


class TestExtractiveGenerator:
    def setup_method(self):
        self.gen = ExtractiveGenerator(min_retrieval_score=0.5)

    def test_empty_chunks(self):
        result = self.gen.generate("What is India?", [])
        assert result is None

    def test_low_score_returns_none(self):
        chunks = [Chunk(chunk_id="c1", document_id="d", text="India is a country.", score=0.3)]
        result = self.gen.generate("What is India?", chunks)
        assert result is None

    def test_high_score_extracts_answer(self):
        chunks = [Chunk(
            chunk_id="c1", document_id="d",
            text="India is a country in South Asia. The capital of India is New Delhi.",
            score=0.95,
        )]
        result = self.gen.generate("What is the capital of India?", chunks)
        assert result is not None
        assert result.method == "extractive"
        assert result.confidence > 0

    def test_citations_included(self):
        chunks = [
            Chunk(chunk_id="c1", document_id="d", text="India is great.", score=0.95),
            Chunk(chunk_id="c2", document_id="d", text="New Delhi is the capital.", score=0.88),
        ]
        result = self.gen.generate("Tell me about India", chunks)
        assert result is not None
        assert len(result.citations) == 2

    def test_citations_truncated(self):
        long_text = "x" * 300
        chunks = [Chunk(chunk_id="c1", document_id="d", text=long_text, score=0.95)]
        result = self.gen.generate("test", chunks)
        assert result is not None
        assert len(result.citations[0].text) <= 200

    def test_hindi_text(self):
        chunks = [Chunk(
            chunk_id="h1", document_id="d",
            text="भारत एक महान देश है। इसकी राजधानी नई दिल्ली है।",
            score=0.93, metadata={"language": "hi"},
        )]
        result = self.gen.generate("भारत की राजधानी क्या है?", chunks)
        assert result is not None
        assert result.method == "extractive"

    def test_gujarati_text(self):
        chunks = [Chunk(
            chunk_id="g1", document_id="d",
            text="ભારત એક મહાન દેશ છે. તેની રાજધાની નવી દિલ્હી છે.",
            score=0.91, metadata={"language": "gu"},
        )]
        result = self.gen.generate("ભારતની રાજધાની શું છે?", chunks)
        assert result is not None
        assert result.method == "extractive"

    def test_confidence_equals_score(self):
        chunks = [Chunk(chunk_id="c1", document_id="d", text="test passage", score=0.75)]
        result = self.gen.generate("test", chunks)
        assert result is not None
        assert result.confidence == 0.75

    def test_min_score_boundary(self):
        gen = ExtractiveGenerator(min_retrieval_score=0.5)
        chunks = [Chunk(chunk_id="c1", document_id="d", text="test passage text", score=0.5)]
        result = gen.generate("test", chunks)
        assert result is not None

    def test_custom_min_score(self):
        gen = ExtractiveGenerator(min_retrieval_score=0.9)
        chunks = [Chunk(chunk_id="c1", document_id="d", text="test", score=0.85)]
        result = gen.generate("test", chunks)
        assert result is None
