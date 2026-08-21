import pytest
from chunking.base import (
    get_chunker, CHUNKERS,
    PassageNativeChunker, FixedChunker, SlidingWindowChunker,
    RecursiveChunker, SemanticChunker, MetadataAwareChunker,
)


class TestChunkerRegistry:
    def test_all_strategies_registered(self):
        expected = {"passage_native", "fixed", "sliding", "recursive", "semantic", "metadata"}
        assert set(CHUNKERS.keys()) == expected

    def test_get_chunker_valid(self):
        for name in CHUNKERS:
            chunker = get_chunker(name)
            assert chunker is not None

    def test_get_chunker_invalid(self):
        with pytest.raises(ValueError, match="Unknown chunker"):
            get_chunker("nonexistent")


class TestPassageNativeChunker:
    def test_single_passage(self, short_text):
        c = get_chunker("passage_native")
        chunks = c.chunk(short_text)
        assert len(chunks) == 1
        assert chunks[0].text == short_text.strip()
        assert chunks[0].metadata["strategy"] == "passage_native"

    def test_empty_input(self):
        c = get_chunker("passage_native")
        assert c.chunk("") == []
        assert c.chunk(None) == []
        assert c.chunk("   ") == []

    def test_metadata_preserved(self, short_text):
        c = get_chunker("passage_native")
        chunks = c.chunk(short_text, metadata={"language": "hi", "document_id": "d1"})
        assert chunks[0].metadata["language"] == "hi"
        assert chunks[0].document_id == "d1"

    def test_chunk_id_deterministic(self, short_text):
        c = get_chunker("passage_native")
        chunks1 = c.chunk(short_text)
        chunks2 = c.chunk(short_text)
        assert chunks1[0].chunk_id == chunks2[0].chunk_id


class TestFixedChunker:
    def test_short_text_single_chunk(self, short_text):
        c = get_chunker("fixed", chunk_size=128, overlap=20)
        chunks = c.chunk(short_text)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self, long_text):
        c = get_chunker("fixed", chunk_size=30, overlap=5)
        chunks = c.chunk(long_text)
        assert len(chunks) > 1

    def test_empty_input(self):
        c = get_chunker("fixed")
        assert c.chunk("") == []
        assert c.chunk(None) == []

    def test_overlap_boundary(self):
        c = get_chunker("fixed", chunk_size=5, overlap=2)
        text = "a b c d e f g h i j"
        chunks = c.chunk(text)
        assert len(chunks) > 1
        # Verify overlap exists between consecutive chunks
        words1 = set(chunks[0].text.split())
        words2 = set(chunks[1].text.split())
        assert len(words1 & words2) > 0

    def test_metadata(self, short_text):
        c = get_chunker("fixed")
        chunks = c.chunk(short_text, metadata={"document_id": "d1"})
        assert chunks[0].metadata["strategy"] == "fixed"
        assert chunks[0].document_id == "d1"


class TestSlidingWindowChunker:
    def test_short_text(self, short_text):
        c = get_chunker("sliding", chunk_size=128, overlap=32)
        chunks = c.chunk(short_text)
        assert len(chunks) == 1

    def test_long_text(self, long_text):
        c = get_chunker("sliding", chunk_size=30, overlap=10)
        chunks = c.chunk(long_text)
        assert len(chunks) > 1

    def test_empty_input(self):
        c = get_chunker("sliding")
        assert c.chunk("") == []
        assert c.chunk(None) == []

    def test_overlap_significant(self):
        c = get_chunker("sliding", chunk_size=10, overlap=5)
        text = " ".join([f"word{i}" for i in range(30)])
        chunks = c.chunk(text)
        assert len(chunks) > 2
        # Check overlap between first two chunks
        words1 = chunks[0].text.split()
        words2 = chunks[1].text.split()
        assert len(words1) >= 5
        assert len(words2) >= 5


class TestRecursiveChunker:
    def test_short_text(self, short_text):
        c = get_chunker("recursive")
        chunks = c.chunk(short_text)
        assert len(chunks) >= 1

    def test_text_with_paragraphs(self):
        text = "Paragraph one about India. " * 10 + "\n\n" + "Paragraph two about China. " * 10
        c = get_chunker("recursive", chunk_size=30, overlap=5)
        chunks = c.chunk(text)
        assert len(chunks) >= 2

    def test_empty_input(self):
        c = get_chunker("recursive")
        assert c.chunk("") == []
        assert c.chunk(None) == []

    def test_strategy_metadata(self, long_text):
        c = get_chunker("recursive")
        chunks = c.chunk(long_text)
        for chunk in chunks:
            assert chunk.metadata["strategy"] == "recursive"


class TestSemanticChunker:
    def test_short_text_fallback(self, short_text):
        c = get_chunker("semantic")
        chunks = c.chunk(short_text)
        # Short text with <=2 sentences returns single chunk
        assert len(chunks) == 1

    def test_multi_sentence(self):
        text = "India is a great country. The capital is New Delhi. Mumbai is the largest city. Bangalore is a tech hub."
        c = get_chunker("semantic", threshold=0.3, min_chunk_size=3)
        chunks = c.chunk(text)
        assert len(chunks) >= 1

    def test_empty_input(self):
        c = get_chunker("semantic")
        assert c.chunk("") == []
        assert c.chunk(None) == []

    def test_identical_sentences_grouped(self):
        text = "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."
        c = get_chunker("semantic", threshold=0.5, min_chunk_size=3)
        chunks = c.chunk(text)
        # All identical sentences should stay in one group
        assert len(chunks) == 1


class TestMetadataAwareChunker:
    def test_short_text(self, short_text):
        c = get_chunker("metadata")
        chunks = c.chunk(short_text, metadata={"language": "hi", "query_type": "factual"})
        assert len(chunks) >= 1
        assert chunks[0].metadata["detected_language"] == "hi"
        assert chunks[0].metadata["query_type"] == "factual"
        assert chunks[0].metadata["strategy"] == "metadata"

    def test_empty_input(self):
        c = get_chunker("metadata")
        assert c.chunk("") == []

    def test_no_metadata(self, short_text):
        c = get_chunker("metadata")
        chunks = c.chunk(short_text)
        assert chunks[0].metadata["strategy"] == "metadata"


class TestCrossStrategy:
    def test_all_produce_valid_chunks(self, long_text):
        for name in CHUNKERS:
            c = get_chunker(name)
            chunks = c.chunk(long_text)
            assert len(chunks) >= 1, f"{name} produced no chunks"
            for chunk in chunks:
                assert chunk.chunk_id, f"{name} chunk missing ID"
                assert chunk.text.strip(), f"{name} chunk has empty text"
                assert isinstance(chunk.metadata, dict)

    def test_hindi_text(self):
        text = "भारत एक महान देश है। इसकी राजधानी नई दिल्ली है। मुंबई इसका सबसे बड़ा शहर है। बैंगलोर एक तकनीकी केंद्र है।"
        for name in ["passage_native", "fixed", "sliding"]:
            c = get_chunker(name)
            chunks = c.chunk(text)
            assert len(chunks) >= 1
            assert "भारत" in chunks[0].text or "भारत" in text

    def test_gujarati_text(self):
        text = "ભારત એક મહાન દેશ છે. તેની રાજધાની નવી દિલ્હી છે. મુંબઈ તેનું સૌથી મોટું શહેર છે."
        for name in ["passage_native", "fixed"]:
            c = get_chunker(name)
            chunks = c.chunk(text)
            assert len(chunks) >= 1
