"""Score a candidate system prompt against the Session-5 Prompt Evaluator.

Usage:
    uv run python evaluate_prompt.py v1    # → JSON scorecard for the draft
    uv run python evaluate_prompt.py v2    # → JSON scorecard for the production prompt

Reads:
  - The evaluator block (between <!--EVALUATOR_START--> and <!--EVALUATOR_END-->)
    from prompt_qualifier.md
  - V1 from the constant below (the deliberately-weak draft)
  - V2 from talk2mcp.SYSTEM_PROMPT_TEMPLATE (the production prompt)
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from llm import LLMClient, detect_provider, resolve_model_name

import talk2mcp


DEFAULT_MODEL = "gemini-3-flash-preview"


# Deliberately-weak draft. Same text lives in README.md.
V1_PROMPT = """You are Inquire, a research assistant that answers questions by chaining Wikipedia lookups.

Tools available:
{tool_spec}

To answer:
1. Break the question into sub-questions.
2. For each sub-question, search Wikipedia, then fetch a summary, then read it.
3. Combine the facts to answer the original question.
4. If your final answer involves arithmetic, use the calculate tool.

You have at most {max_iter} turns.

Each turn, output a JSON object with "thought" and either "tool" (with name and args)
or "final_answer".
"""


def load_evaluator() -> str:
    """Extract the evaluator block from prompt_qualifier.md."""
    qfile = Path(__file__).parent / "prompt_qualifier.md"
    text = qfile.read_text(encoding="utf-8")
    m = re.search(r"<!--EVALUATOR_START-->(.*?)<!--EVALUATOR_END-->", text, re.DOTALL)
    if not m:
        raise SystemExit("Could not find EVALUATOR markers in prompt_qualifier.md")
    return m.group(1).strip()


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced `{...}` substring, or None if none found."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


async def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("v1", "v2"):
        print("usage: uv run python evaluate_prompt.py v1|v2", file=sys.stderr)
        return 2
    label = sys.argv[1]

    load_dotenv()
    provider = detect_provider()
    model_name = resolve_model_name(provider, default=DEFAULT_MODEL)
    llm = LLMClient(provider=provider, model_name=model_name)

    evaluator = load_evaluator()
    target = V1_PROMPT if label == "v1" else talk2mcp.SYSTEM_PROMPT_TEMPLATE

    # Wrapper directive — the evaluator block itself stays verbatim. Small models
    # (gemini-3-flash-preview) need the schema repeated at the END for the keys
    # to stay anchored; otherwise they invent their own field names.
    prompt = (
        f"{evaluator}\n\n"
        "---\nPROMPT TO EVALUATE:\n\n"
        f"{target}\n\n"
        "---\nOUTPUT INSTRUCTIONS:\n"
        "Respond with ONLY a single JSON object using EXACTLY these 9 keys (do not "
        "invent new keys, do not rename existing ones, do not omit any):\n"
        '  "explicit_reasoning"        boolean   (criterion 1)\n'
        '  "structured_output"         boolean   (criterion 2)\n'
        '  "tool_separation"           boolean   (criterion 3)\n'
        '  "conversation_loop"         boolean   (criterion 4)\n'
        '  "instructional_framing"     boolean   (criterion 5)\n'
        '  "internal_self_checks"      boolean   (criterion 6)\n'
        '  "reasoning_type_awareness"  boolean   (criterion 7)\n'
        '  "fallbacks"                 boolean   (criterion 8)\n'
        '  "overall_clarity"           string    (criterion 9 — one sentence)\n'
        "No markdown fences, no prose around the JSON, no other keys."
    )

    print(f"[evaluating] {label} ({len(target)} chars)", file=sys.stderr)
    print(f"[model]      {model_name} ({provider})", file=sys.stderr)

    raw, _ = await llm.generate(prompt)

    obj_text = _extract_first_json_object(raw)
    if obj_text:
        try:
            parsed = json.loads(obj_text)
            print(json.dumps(parsed, indent=2))
            return 0
        except json.JSONDecodeError as e:
            print(f"[warn] JSON parse failed: {e}", file=sys.stderr)

    # Fallback: dump raw model output.
    print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
