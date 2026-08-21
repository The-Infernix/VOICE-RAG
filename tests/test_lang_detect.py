import pytest
from retrieval.lang_detect import detect_query_language


class TestDetectQueryLanguage:
    def test_english(self):
        assert detect_query_language("Who is the Prime Minister of India?") == "en"

    def test_hindi_devanagari(self):
        assert detect_query_language("भारत की राजधानी क्या है?") == "hi"

    def test_gujarati(self):
        assert detect_query_language("ભારતની રાજધાની શું છે?") == "gu"

    def test_empty(self):
        assert detect_query_language("") == "en"
        assert detect_query_language(None) == "en"

    def test_mixed_script_gujarati_wins(self):
        assert detect_query_language("hello ભારત") == "gu"

    def test_mixed_script_hindi(self):
        assert detect_query_language("hello भारत") == "hi"
