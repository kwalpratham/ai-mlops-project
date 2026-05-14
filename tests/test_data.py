"""Tests for the data pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import clean_text


class TestCleanText:
    def test_removes_html_tags(self):
        assert clean_text("Hello <b>world</b>") == "Hello world"

    def test_removes_urls(self):
        assert clean_text("Visit http://example.com today") == "Visit today"

    def test_normalizes_whitespace(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_combined_cleaning(self):
        raw = '  Check <a href="http://x.com">this</a>  out  '
        result = clean_text(raw)
        assert "http" not in result
        assert "<" not in result
        assert "  " not in result
