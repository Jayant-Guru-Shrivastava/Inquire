"""Pydantic input models for every Inquire MCP tool.

Single source of truth: these models validate args inside `mcp_server.py`
and (transitively, via the FastMCP tool decorator) drive the JSON Schema
that the agent's system prompt renders.

Session 5's "Pydantic on every boundary" rule — applied here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ReasoningType = Literal[
    "Decomposition",
    "Lookup",
    "Inference",
    "Verification",
    "Synthesis",
    "Computation",
]


class ReasoningStep(BaseModel):
    """One reasoning step in the agent's chain. Logged, no I/O."""

    step: int = Field(ge=1, le=30, description="Sequential step number (1..30)")
    reasoning_type: ReasoningType = Field(description="Tag the kind of reasoning this step performs")
    text: str = Field(min_length=1, max_length=600, description="One sentence on what you're reasoning about")


class WikiSearch(BaseModel):
    """Search English Wikipedia for matching page titles."""

    query: str = Field(min_length=1, max_length=200, description="Free-text query")
    limit: int = Field(default=5, ge=1, le=10, description="Max number of hits to return")


class WikiFetch(BaseModel):
    """Fetch the lead-section summary of a specific Wikipedia page."""

    title: str = Field(min_length=1, max_length=200, description="Exact Wikipedia page title")
    max_chars: int = Field(default=2000, ge=200, le=8000, description="Max characters of extract to return")


class Calc(BaseModel):
    """Evaluate a numeric expression (digits + - * / ( ) . and whitespace only)."""

    expression: str = Field(min_length=1, max_length=200, description="Arithmetic expression, e.g. '975000 - 700000'")


class Verify(BaseModel):
    """Self-check: does the evidence passage support the claim?"""

    claim: str = Field(min_length=1, max_length=400, description="The factual claim to verify")
    evidence: str = Field(min_length=1, max_length=8000, description="Passage that should support or contradict the claim")
