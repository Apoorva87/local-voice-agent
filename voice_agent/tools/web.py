"""Web search via DuckDuckGo's HTML endpoint.

No API key and no account, which keeps the "free and open components"
constraint intact. This is the one part of the system that leaves the
machine, and only when the user asks a question that needs it.

Results are compacted hard: the model is about to speak them aloud, so a
handful of short snippets beats a page of text.
"""

from __future__ import annotations

import asyncio
import html
import re
import time

import httpx
from loguru import logger

SEARCH_URL = "https://html.duckduckgo.com/html/"
# Web search is the slowest tool in the system; past this the agent should
# apologise rather than keep the user waiting in silence.
TIMEOUT_SECS = 8.0
MAX_RESULTS = 4
MAX_SNIPPET_CHARS = 220

# Titles and snippets are matched separately and zipped. A single combined
# pattern is fragile here: the two anchors sit far apart in the markup, and an
# optional lazy group for the snippet matches empty every time.
_TITLE_RE = re.compile(r'<a[^>]*class="result__a"[^>]*>(.*?)</a>', re.DOTALL)
_SNIPPET_RE = re.compile(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(fragment: str | None) -> str:
    if not fragment:
        return ""
    return html.unescape(_TAG_RE.sub("", fragment)).strip()


def parse_results(markup: str, limit: int = MAX_RESULTS) -> list[tuple[str, str]]:
    """Extract (title, snippet) pairs from DuckDuckGo's HTML."""
    titles = [_clean(t) for t in _TITLE_RE.findall(markup)]
    snippets = [_clean(s) for s in _SNIPPET_RE.findall(markup)]

    results: list[tuple[str, str]] = []
    for index, title in enumerate(titles):
        if not title:
            continue
        snippet = snippets[index] if index < len(snippets) else ""
        if len(snippet) > MAX_SNIPPET_CHARS:
            snippet = snippet[:MAX_SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."
        results.append((title, snippet))
        if len(results) >= limit:
            break
    return results


async def search(query: str) -> str:
    """Run a search and return a compact, speakable summary."""
    query = (query or "").strip()
    if not query:
        return "No search query was given."

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECS, follow_redirects=True) as client:
            response = await client.post(
                SEARCH_URL,
                data={"q": query},
                headers={
                    # DuckDuckGo returns an empty page to clients with no
                    # recognisable user agent.
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()
    except asyncio.CancelledError:
        # Expected when the user interrupts; not an error worth reporting.
        raise
    except httpx.TimeoutException:
        logger.warning(f"Web search timed out for {query!r}")
        return "The web search took too long. Tell the user you could not reach the web just now."
    except Exception as exc:
        logger.warning(f"Web search failed for {query!r}: {exc}")
        return f"The web search failed: {exc}"

    results = parse_results(response.text)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"Web search {query!r} -> {len(results)} results in {elapsed:.0f}ms")

    if not results:
        return f"No results were found for {query}."

    lines = [f"{title}. {snippet}".strip() for title, snippet in results]
    return "Search results:\n" + "\n".join(f"- {line}" for line in lines)
