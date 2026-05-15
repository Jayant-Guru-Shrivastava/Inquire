"""Shared LLM client for SlideAgent.

Imported by both `talk2mcp.py` (the agent loop) and `mcp_server.py` (the
review_deck tool needs vision calls). Loads .env automatically so either side
gets the same auth config.

Provider priority: OpenRouter > Vertex AI > AI Studio.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI


# Module-global throttle state. Each process (client / server) has its own; for
# the agent's main loop and the server's review_deck call to be coordinated we
# would need IPC. In practice review_deck fires once near the end, so cross-
# process throttle skew is acceptable.
_last_call_at: float = 0.0


async def respect_rate_limit() -> None:
    """Sleep just enough to honour GEMINI_MIN_INTERVAL_SECONDS between LLM calls."""
    global _last_call_at
    try:
        min_interval = float(os.getenv("GEMINI_MIN_INTERVAL_SECONDS", "12.5"))
    except ValueError:
        min_interval = 12.5
    if min_interval <= 0:
        return
    now = time.monotonic()
    elapsed = now - _last_call_at
    if elapsed < min_interval:
        wait = min_interval - elapsed
        print(f"[rate-limit] sleeping {wait:.1f}s to stay under quota", flush=True)
        await asyncio.sleep(wait)


def _mark_call() -> None:
    global _last_call_at
    _last_call_at = time.monotonic()


def total_tokens(usage: Any) -> int:
    """Normalise usage from google-genai or OpenAI SDKs to a single integer."""
    if usage is None:
        return 0
    for attr in ("total_token_count", "total_tokens"):
        v = getattr(usage, attr, None)
        if isinstance(v, int):
            return v
    return 0


# ---------------------------------------------------------------------------
# Provider detection + factory
# ---------------------------------------------------------------------------

def detect_provider() -> str:
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes"):
        return "vertex"
    if os.getenv("GEMINI_API_KEY"):
        return "aistudio"
    raise SystemExit(
        "ERROR: No auth configured. Set one of: OPENROUTER_API_KEY (+ OPENROUTER_MODEL), "
        "GEMINI_API_KEY (AI Studio), or GOOGLE_GENAI_USE_VERTEXAI=true (+ GOOGLE_CLOUD_PROJECT)."
    )


def resolve_model_name(provider: str, default: str = "gemini-3-flash-preview") -> str:
    if provider == "openrouter":
        return os.getenv("OPENROUTER_MODEL") or "google/gemini-3-flash-preview"
    return os.getenv("GEMINI_MODEL", default)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Provider-agnostic Gemini wrapper. Supports text and image inputs."""

    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        if provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise SystemExit("OPENROUTER_API_KEY is missing.")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            self.genai_client = None
        else:
            self.openai_client = None
            if provider == "vertex":
                project = os.getenv("GOOGLE_CLOUD_PROJECT")
                location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
                if not project:
                    raise SystemExit("Vertex requires GOOGLE_CLOUD_PROJECT.")
                self.genai_client = genai.Client(vertexai=True, project=project, location=location)
            else:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise SystemExit("GEMINI_API_KEY is missing.")
                self.genai_client = genai.Client(api_key=api_key)

    def _openrouter_headers(self) -> dict:
        return {
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "https://github.com/Jayant-Guru-Shrivastava/Talk2MCP"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "SlideAgent"),
        }

    async def generate(self, prompt: str) -> tuple[str, Any]:
        """Plain text generation."""
        await respect_rate_limit()
        if self.provider == "openrouter":
            resp = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096,
                extra_headers=self._openrouter_headers(),
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
        else:
            response = await self.genai_client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3, max_output_tokens=4096
                ),
            )
            text = response.text or ""
            usage = getattr(response, "usage_metadata", None)
        _mark_call()
        return text, usage

    async def generate_with_images(self, prompt: str, image_paths: list[str]) -> tuple[str, Any]:
        """Multimodal generation. Sends `prompt` plus the named image files."""
        await respect_rate_limit()
        if self.provider == "openrouter":
            content: list[dict] = [{"type": "text", "text": prompt}]
            for path in image_paths:
                mime = _mime_for(path)
                b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            resp = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0.3,
                max_tokens=4096,
                extra_headers=self._openrouter_headers(),
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
        else:
            parts: list[Any] = [genai_types.Part.from_text(text=prompt)]
            for path in image_paths:
                mime = _mime_for(path)
                parts.append(
                    genai_types.Part.from_bytes(data=Path(path).read_bytes(), mime_type=mime)
                )
            response = await self.genai_client.aio.models.generate_content(
                model=self.model_name,
                contents=parts,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3, max_output_tokens=4096
                ),
            )
            text = response.text or ""
            usage = getattr(response, "usage_metadata", None)
        _mark_call()
        return text, usage


def _mime_for(path: str) -> str:
    p = path.lower()
    if p.endswith(".png"):
        return "image/png"
    if p.endswith(".webp"):
        return "image/webp"
    if p.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def build_llm_from_env() -> LLMClient:
    """Convenience factory: loads .env, detects provider, returns ready LLMClient."""
    load_dotenv()
    provider = detect_provider()
    model = resolve_model_name(provider)
    return LLMClient(provider=provider, model_name=model)
