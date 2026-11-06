"""
X-Star Inference Web Browser Tool (/browser Slash Command)
==========================================================
Enables real-time web browsing, factual ground truth retrieval, and URL reading
directly within the X-Star inference CLI.

Supports:
- Web Search (DuckDuckGo Instant Answer + Wikipedia Knowledge API)
- Direct URL fetching, HTML cleaning, and textual distillation
- Context packaging for injection into LLM multi-turn conversation
"""

from __future__ import annotations

import re
import html
import urllib.parse
from typing import List, Dict, Any, Optional
import requests


class WebBrowser:
    """
    Lightweight, dependency-free web browsing and factual search assistant
    for grounding X-Star inference sessions.
    """

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

    @staticmethod
    def is_url(text: str) -> bool:
        """Determines if input string is an HTTP/HTTPS URL."""
        text = text.strip()
        return bool(re.match(r"^https?://[^\s]+$", text, re.IGNORECASE))

    @classmethod
    def clean_html(cls, raw_html: str) -> str:
        """Strips HTML tags, scripts, and normalizes whitespace."""
        # 1. Remove script and style elements
        cleaned = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw_html)
        # 2. Replace breaks and block tags with newlines
        cleaned = re.sub(r"(?i)<(p|br|div|li|h[1-6]|tr)[^>]*>", "\n", cleaned)
        # 3. Strip all remaining tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)
        # 4. Unescape HTML entities
        cleaned = html.unescape(cleaned)
        # 5. Normalize whitespace
        lines = [line.strip() for line in cleaned.splitlines()]
        cleaned = "\n".join(l for l in lines if l)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned.strip()

    @classmethod
    def fetch_url(cls, url: str, timeout: int = 8, max_chars: int = 3500) -> Dict[str, Any]:
        """
        Fetches webpage content from URL and returns title, extracted text, and status.
        """
        try:
            headers = {"User-Agent": cls.USER_AGENT}
            resp = requests.get(url, headers=headers, timeout=timeout, verify=True)
            resp.raise_for_status()

            raw_html = resp.text
            # Extract title
            title_match = re.search(r"(?i)<title[^>]*>(.*?)</title>", raw_html)
            title = html.unescape(title_match.group(1).strip()) if title_match else url

            content = cls.clean_html(raw_html)
            if len(content) > max_chars:
                content = content[:max_chars] + "\n... [Content truncated for context window]"

            return {
                "status": "success",
                "url": url,
                "title": title,
                "content": content,
                "status_code": resp.status_code,
            }
        except Exception as e:
            return {
                "status": "error",
                "url": url,
                "title": url,
                "content": f"Failed to retrieve URL {url}: {str(e)}",
                "error": str(e),
            }

    @classmethod
    def search(cls, query: str, max_results: int = 4, timeout: int = 6) -> List[Dict[str, str]]:
        """
        Executes web search across DuckDuckGo API and Wikipedia Knowledge API.
        Returns aggregated list of {'title', 'snippet', 'url'}.
        """
        results: List[Dict[str, str]] = []
        encoded_query = urllib.parse.quote_plus(query)

        # 1. Wikipedia Knowledge API
        try:
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&utf8=&format=json"
            wiki_resp = requests.get(wiki_url, headers={"User-Agent": cls.USER_AGENT}, timeout=timeout)
            if wiki_resp.status_code == 200:
                wiki_data = wiki_resp.json()
                search_items = wiki_data.get("query", {}).get("search", [])
                for item in search_items[:max_results]:
                    title = item.get("title", "")
                    raw_snippet = item.get("snippet", "")
                    clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet)
                    clean_snippet = html.unescape(clean_snippet).strip()
                    page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                    results.append({
                        "title": title,
                        "snippet": clean_snippet,
                        "url": page_url,
                        "source": "Wikipedia",
                    })
        except Exception:
            pass

        # 2. DuckDuckGo Instant Answer API
        try:
            ddg_url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
            ddg_resp = requests.get(ddg_url, headers={"User-Agent": cls.USER_AGENT}, timeout=timeout)
            if ddg_resp.status_code in (200, 202):
                ddg_data = ddg_resp.json()
                abstract = ddg_data.get("AbstractText", "") or ddg_data.get("Abstract", "")
                abstract_url = ddg_data.get("AbstractURL", "")
                abstract_source = ddg_data.get("AbstractSource", "DuckDuckGo")
                if abstract:
                    results.insert(0, {
                        "title": ddg_data.get("Heading", query),
                        "snippet": abstract,
                        "url": abstract_url or "https://duckduckgo.com/?q=" + encoded_query,
                        "source": abstract_source,
                    })

                # Related topics
                related = ddg_data.get("RelatedTopics", [])
                for item in related:
                    if isinstance(item, dict) and "Text" in item and len(results) < max_results + 2:
                        results.append({
                            "title": item.get("Text", "").split(" - ")[0],
                            "snippet": item.get("Text", ""),
                            "url": item.get("FirstURL", ""),
                            "source": "DuckDuckGo",
                        })
        except Exception:
            pass

        return results[:max_results]

    @classmethod
    def format_search_context(cls, query: str, results: List[Dict[str, str]]) -> str:
        """Formats search results as an injected context block for the conversation."""
        if not results:
            return f"[WEB BROWSING]: No external search results found for query: '{query}'."

        formatted = [f"=== WEB BROWSING EVIDENCE: '{query}' ==="]
        for i, res in enumerate(results, start=1):
            formatted.append(f"[{i}] {res['title']} ({res.get('source', 'Web')})")
            formatted.append(f"    URL: {res.get('url', '')}")
            formatted.append(f"    Summary: {res.get('snippet', '')}\n")
        formatted.append("=== END WEB BROWSING EVIDENCE ===")
        return "\n".join(formatted)

    @classmethod
    def format_url_context(cls, page_data: Dict[str, Any]) -> str:
        """Formats page fetch results as an injected context block."""
        return (
            f"=== BROWSED WEBPAGE: {page_data.get('title', 'Document')} ===\n"
            f"URL: {page_data.get('url', '')}\n"
            f"Content:\n{page_data.get('content', '')}\n"
            f"=== END BROWSED WEBPAGE ==="
        )
