"""Inquire MCP server — 5 tools for multi-hop Wikipedia research.

Tools:
  1. show_reasoning         — log a reasoning step (audit trail)
  2. search_wikipedia       — find matching page titles
  3. fetch_wikipedia_summary — get the lead-section extract
  4. calculate              — safe arithmetic for numeric follow-ups
  5. verify_claim           — keyword-grounded self-check

Every tool returns {"ok": bool, ...}. Bad args fail validation via Pydantic
(see models.py) and return {"ok": False, "error": "..."} rather than raising.
Runs over stdio; spawned by talk2mcp.py.
"""

from __future__ import annotations

import re

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

import wiki
from models import Calc, ReasoningStep, Verify, WikiFetch, WikiSearch


mcp = FastMCP("Inquire")


# ---------------------------------------------------------------------------
# Tool 1 — show_reasoning  (audit log; no I/O beyond stdout)
# ---------------------------------------------------------------------------

@mcp.tool()
def show_reasoning(step: int, reasoning_type: str, text: str) -> dict:
    """Log a reasoning step. Call this BEFORE every external lookup.

    reasoning_type must be one of: Decomposition, Lookup, Inference,
    Verification, Synthesis, Computation. The runtime prints the step
    in the iteration log so the chain of thought is auditable.

    No side-effects beyond logging — but the agent is REQUIRED by the
    qualified system prompt to emit one before each tool call.
    """
    try:
        s = ReasoningStep(step=step, reasoning_type=reasoning_type, text=text)
    except ValidationError as e:
        return {"ok": False, "error": f"bad reasoning step: {e}"}
    return {
        "ok": True,
        "logged": True,
        "step": s.step,
        "reasoning_type": s.reasoning_type,
    }


# ---------------------------------------------------------------------------
# Tool 2 — search_wikipedia
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_wikipedia(query: str, limit: int = 5) -> dict:
    """Search English Wikipedia for pages matching `query`.

    Returns up to `limit` hits as {title, snippet, page_id}. An empty
    `results` list is NOT an error — it means rephrase the query and try
    again. Network failures also return ok:True with an empty list so the
    agent can keep going.
    """
    try:
        args = WikiSearch(query=query, limit=limit)
    except ValidationError as e:
        return {"ok": False, "error": f"bad input: {e}"}
    results = await wiki.search(args.query, args.limit)
    return {
        "ok": True,
        "query": args.query,
        "results": results,
        "total": len(results),
    }


# ---------------------------------------------------------------------------
# Tool 3 — fetch_wikipedia_summary
# ---------------------------------------------------------------------------

@mcp.tool()
async def fetch_wikipedia_summary(title: str, max_chars: int = 2000) -> dict:
    """Fetch the lead-section summary of a specific Wikipedia page.

    On success: {ok, title, page_url, extract, truncated, full_chars}.
    On 404 / network failure: {ok:False, error:"..."}.

    If `truncated` is true the extract was cut to `max_chars`. Re-call with a
    larger max_chars if the answer might be later in the article.
    """
    try:
        args = WikiFetch(title=title, max_chars=max_chars)
    except ValidationError as e:
        return {"ok": False, "error": f"bad input: {e}"}
    return await wiki.summary(args.title, args.max_chars)


# ---------------------------------------------------------------------------
# Tool 4 — calculate  (safe arithmetic)
# ---------------------------------------------------------------------------

_SAFE_EXPR = re.compile(r"^[\d\s\+\-\*/\(\)\.]+$")


@mcp.tool()
def calculate(expression: str) -> dict:
    """Evaluate a numeric expression safely.

    Allowed characters: digits, whitespace, `+ - * / ( ) .` and commas
    (commas are stripped as thousands separators). Anything else returns
    {ok:False, error}. Use this for steps like "difference in population"
    or "ratio of X to Y".
    """
    try:
        args = Calc(expression=expression)
    except ValidationError as e:
        return {"ok": False, "error": f"bad input: {e}"}

    expr = args.expression.replace(",", "")
    if not _SAFE_EXPR.match(expr):
        return {
            "ok": False,
            "error": (
                "expression contains disallowed characters. Allowed: digits, "
                "whitespace, + - * / ( ) . and commas (as thousands separators)."
            ),
        }
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — sandboxed by regex above
    except (SyntaxError, ZeroDivisionError, TypeError, ValueError) as e:
        return {"ok": False, "error": f"could not evaluate: {e}"}
    if not isinstance(result, (int, float)):
        return {"ok": False, "error": f"result is not numeric: {result!r}"}
    return {"ok": True, "expression": args.expression, "result": float(result)}


# ---------------------------------------------------------------------------
# Tool 5 — verify_claim  (keyword-grounded self-check)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "these", "those",
    "have", "has", "had", "was", "were", "are", "is", "been", "being", "but",
    "not", "they", "their", "there", "than", "then", "into", "onto", "upon",
    "also", "very", "more", "most", "some", "such", "only", "what", "when",
    "where", "which", "while", "would", "could", "should", "between",
    "during", "after", "before", "about", "against", "without", "within",
    "would", "shall", "must", "your", "yours", "ours", "them", "him", "her",
}


def _significant_tokens(text: str) -> list[str]:
    """Lowercase tokens length ≥ 4 that aren't English stop-words."""
    return [
        w for w in re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
        if len(w) >= 4 and w not in _STOP_WORDS
    ]


def _token_in_evidence(token: str, evidence_lower: str) -> bool:
    """Match `token` to `evidence_lower` with cheap inflection-tolerance.

    A claim token counts as matched if itself appears literally OR a 4+-char
    prefix of it appears. This lets "invented" (claim) match "inventor"
    (evidence) — both contain the prefix "invent" — without any stemming
    library. Asymmetric truncation also catches the reverse: a claim token
    "invent" is found as a substring of "inventor" by the literal check.
    """
    if token in evidence_lower:
        return True
    for cut in (1, 2, 3):
        if len(token) - cut >= 4 and token[:-cut] in evidence_lower:
            return True
    return False


@mcp.tool()
def verify_claim(claim: str, evidence: str) -> dict:
    """Check whether `evidence` supports `claim` (keyword-grounded, deterministic).

    Splits the claim into significant tokens (length ≥ 4, not stop-words) and
    counts how many appear in `evidence` (case-insensitive, with cheap
    inflection-tolerance — "invented" matches "inventor" because both share
    the 6-char prefix "invent").
       ≥ 80% present   → supports="yes"
       40 – 80%        → supports="partial"
       <  40%          → supports="no"

    Intentionally cheap and explainable — no extra LLM call. The runtime
    requires at least one verify_claim with supports="yes" before final_answer.
    """
    try:
        args = Verify(claim=claim, evidence=evidence)
    except ValidationError as e:
        return {"ok": False, "error": f"bad input: {e}"}

    claim_tokens = _significant_tokens(args.claim)
    if not claim_tokens:
        return {
            "ok": False,
            "error": "claim has no significant tokens to verify (try a more specific claim)",
        }

    evidence_lower = args.evidence.lower()
    hits = [t for t in claim_tokens if _token_in_evidence(t, evidence_lower)]
    ratio = len(hits) / len(claim_tokens)

    if ratio >= 0.8:
        supports = "yes"
    elif ratio >= 0.4:
        supports = "partial"
    else:
        supports = "no"

    return {
        "ok": True,
        "supports": supports,
        "tokens_total": len(claim_tokens),
        "tokens_matched": len(hits),
        "match_ratio": round(ratio, 3),
        "reason": (
            f"matched {len(hits)}/{len(claim_tokens)} significant tokens "
            f"from the claim in the evidence ({supports})"
        ),
    }


if __name__ == "__main__":
    mcp.run()
