"""Async English Wikipedia client — no API key, no login.

Two public coroutines:
    results = await search("Tim Berners-Lee")
        # → [{"title": "...", "snippet": "...", "page_id": "..."}, ...]
    page    = await summary("Tim Berners-Lee")
        # → {"ok": True, "title": ..., "extract": ..., "page_url": ..., "truncated": bool}

Both raise NO exceptions — failures return [] or {"ok": False, "error": "..."}.

In-memory caches prevent the agent from paying for the same lookup twice
within one MCP server lifetime (the server lives for one agent run, then
dies with stdio).
"""

from __future__ import annotations

import urllib.parse

import httpx


WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_REST = "https://en.wikipedia.org/api/rest_v1"
USER_AGENT = (
    "InquireAgent/1.0 "
    "(https://github.com/Jayant-Guru-Shrivastava/Inquire; educational use)"
)
TIMEOUT = httpx.Timeout(15.0)

_search_cache: dict[tuple[str, int], list[dict]] = {}
_summary_cache: dict[str, dict] = {}


async def search(query: str, limit: int = 5) -> list[dict]:
    """Search Wikipedia for matching page titles via the `list=search` action.

    Returns up to `limit` dicts, each with {title, snippet, page_id}.
    Empty list on no results or network failure — never raises.
    """
    q = (query or "").strip()
    if not q:
        return []
    cache_key = (q, int(limit))
    if cache_key in _search_cache:
        return list(_search_cache[cache_key])

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    params = {
        "action": "query",
        "list": "search",
        "srsearch": q,
        "srlimit": max(1, min(int(limit), 10)),
        "format": "json",
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers, follow_redirects=True) as client:
            r = await client.get(WIKI_API, params=params)
            r.raise_for_status()
            hits = r.json().get("query", {}).get("search", [])
    except Exception:
        return []

    results: list[dict] = []
    for h in hits:
        title = h.get("title", "")
        snippet_html = h.get("snippet", "")
        # Strip search-highlight markup that Wikipedia adds.
        snippet = (
            snippet_html
            .replace('<span class="searchmatch">', "")
            .replace("</span>", "")
            .replace("&quot;", '"')
            .replace("&amp;", "&")
        )
        results.append({
            "title": title,
            "snippet": snippet.strip(),
            "page_id": str(h.get("pageid", "")),
        })

    _search_cache[cache_key] = list(results)
    return results


async def summary(title: str, max_chars: int = 2000) -> dict:
    """Fetch the lead-section summary for a Wikipedia page via REST v1.

    Returns {ok, title, page_url, extract, truncated, full_chars} on success;
    {ok: False, error: "..."} on 404 or network failure.
    """
    t = (title or "").strip()
    if not t:
        return {"ok": False, "error": "title is empty"}

    if t in _summary_cache:
        return _apply_truncation(_summary_cache[t], max_chars)

    quoted = urllib.parse.quote(t.replace(" ", "_"), safe="")
    url = f"{WIKI_REST}/page/summary/{quoted}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers, follow_redirects=True) as client:
            r = await client.get(url)
            if r.status_code == 404:
                return {"ok": False, "error": f"page not found: {t}"}
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"wikipedia request failed: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"unexpected wikipedia error: {e}"}

    extract_full = (data.get("extract") or "").strip()
    page_url = (
        data.get("content_urls", {}).get("desktop", {}).get("page")
        or data.get("content_urls", {}).get("mobile", {}).get("page")
        or f"https://en.wikipedia.org/wiki/{quoted}"
    )

    record = {
        "ok": True,
        "title": data.get("title", t),
        "page_url": page_url,
        "extract_full": extract_full,
    }
    _summary_cache[t] = record
    return _apply_truncation(record, max_chars)


def _apply_truncation(record: dict, max_chars: int) -> dict:
    """Return a fresh dict with the extract trimmed to `max_chars`."""
    full = record.get("extract_full", "")
    n = max(200, min(int(max_chars), 8000))
    truncated = len(full) > n
    extract = full[:n] + ("…" if truncated else "")
    return {
        "ok": record.get("ok", False),
        "title": record.get("title"),
        "page_url": record.get("page_url"),
        "extract": extract,
        "truncated": truncated,
        "full_chars": len(full),
    }


# Tiny CLI for one-off testing: `python wiki.py "Tim Berners-Lee"`.
if __name__ == "__main__":
    import asyncio
    import sys

    async def _demo() -> None:
        q = " ".join(sys.argv[1:]) or "Tim Berners-Lee"
        print(f"--- search('{q}') ---")
        hits = await search(q, limit=3)
        for h in hits:
            print(f"  • {h['title']} :: {h['snippet'][:80]}…")
        if hits:
            print(f"\n--- summary('{hits[0]['title']}') ---")
            s = await summary(hits[0]["title"], max_chars=500)
            if s.get("ok"):
                print(s["extract"])
                print(f"\n[{s['full_chars']} chars total, truncated={s['truncated']}]")
            else:
                print(f"(error) {s.get('error')}")

    asyncio.run(_demo())
