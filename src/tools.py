from __future__ import annotations
import logging
import requests

logger = logging.getLogger(__name__)

_DDG_URL = "https://api.duckduckgo.com/"
_MAX_RESULTS = 5


def web_search(query: str) -> dict:
    try:
        response = requests.get(
            _DDG_URL,
            params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for topic in data.get("RelatedTopics", [])[:_MAX_RESULTS]:
            if "Text" not in topic or "FirstURL" not in topic:
                continue
            text = topic["Text"]
            dot_idx = text.find(".")
            title = text[:dot_idx].strip() if dot_idx > 0 else text[:60].strip()
            snippet = text[dot_idx + 1:].strip() if dot_idx > 0 else text
            results.append({"title": title, "snippet": snippet, "url": topic["FirstURL"]})

        return {"results": results}
    except requests.exceptions.Timeout:
        logger.warning("DuckDuckGo search timed out for query: %s", query)
        return {"results": [], "error": "Search timed out"}
    except requests.exceptions.RequestException as exc:
        logger.error("DuckDuckGo search request failed: %s", exc)
        return {"results": [], "error": "Search unavailable"}
    except (ValueError, KeyError) as exc:
        logger.error("DuckDuckGo search response parse failed: %s", exc)
        return {"results": [], "error": "Search unavailable"}
