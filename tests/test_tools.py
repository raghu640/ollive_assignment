import pytest
from unittest.mock import patch, MagicMock
from src.tools import web_search

MOCK_DDG_RESPONSE = {
    "RelatedTopics": [
        {"Text": "Python is a programming language. python.org", "FirstURL": "https://python.org"},
        {"Text": "Python 3.12 released. See more at python.org/news", "FirstURL": "https://python.org/news"},
    ]
}


def test_web_search_returns_dict_with_results():
    with patch("src.tools.requests.get") as mock_get:
        mock_get.return_value = MagicMock(json=lambda: MOCK_DDG_RESPONSE, raise_for_status=lambda: None)
        result = web_search("Python programming")
    assert "results" in result
    assert isinstance(result["results"], list)


def test_web_search_result_structure():
    with patch("src.tools.requests.get") as mock_get:
        mock_get.return_value = MagicMock(json=lambda: MOCK_DDG_RESPONSE, raise_for_status=lambda: None)
        result = web_search("Python programming")
    for item in result["results"]:
        assert "title" in item
        assert "snippet" in item
        assert "url" in item


def test_web_search_returns_empty_on_http_error():
    with patch("src.tools.requests.get") as mock_get:
        mock_get.side_effect = Exception("Network error")
        result = web_search("anything")
    assert result == {"results": [], "error": "Search unavailable"}


def test_web_search_limits_to_five_results():
    many_topics = [
        {"Text": f"Result {i}. example.com/{i}", "FirstURL": f"https://example.com/{i}"}
        for i in range(10)
    ]
    with patch("src.tools.requests.get") as mock_get:
        mock_get.return_value = MagicMock(
            json=lambda: {"RelatedTopics": many_topics}, raise_for_status=lambda: None
        )
        result = web_search("test")
    assert len(result["results"]) <= 5
