"""Inquire MCP client.

Spawns mcp_server.py over stdio, asks Gemini to answer a multi-hop research
question by chaining Wikipedia lookups + self-verification, and prints a
cited answer. The full conversation history is included in every LLM call.

Usage:
    uv run python talk2mcp.py "What is the capital of the country where the inventor of the World Wide Web was born?"
    uv run python talk2mcp.py "..."  2>&1 | tee run.log
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

from llm import (
    LLMClient,
    detect_provider,
    resolve_model_name,
    total_tokens as _total_tokens,
)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


DEFAULT_MODEL = "gemini-3-flash-preview"
MAX_ITER = 20
DEFAULT_TOPIC = "What is the capital of the country where the inventor of the World Wide Web was born?"


# ---------------------------------------------------------------------------
# System prompt — the QUALIFIED V2 prompt (V1 + prompt_qualifier.md → this)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are Inquire, a multi-hop research agent. You answer chained questions by decomposing them into sub-questions, looking each one up on Wikipedia, verifying the keystone facts, then synthesising a cited answer.

# Tools (call exactly ONE per turn)
{tool_spec}

# Mandatory sequence
  1.  show_reasoning(step=1, reasoning_type="Decomposition", text="<list the sub-questions in order>")
  2.  For EACH sub-question, in order:
       a. show_reasoning(reasoning_type="Lookup", text="searching for <X>")
       b. search_wikipedia(query="...")
       c. fetch_wikipedia_summary(title="<top result>")
       d. show_reasoning(reasoning_type="Inference", text="from this summary I learned <fact>")
       e. verify_claim(claim="<the fact you just inferred>", evidence="<the extract>")
  3.  If arithmetic is needed: show_reasoning(reasoning_type="Computation", ...) → calculate(...)
  4.  show_reasoning(reasoning_type="Synthesis", text="combining everything: <chain of facts>")
  5.  Emit final_answer — ONLY after at least one verify_claim returned ok=True AND supports="yes".

# Output format — EVERY turn — EXACTLY ONE of these two JSON shapes, nothing else

  A) Tool call:
  {{"thought": "<one short sentence: why this tool now>",
    "reasoning_type": "Decomposition|Lookup|Inference|Verification|Synthesis|Computation",
    "tool": {{"name": "<tool_name>", "args": {{...}}}}}}

  B) Final answer (terminal — emit ONLY after the self-check below passes):
  {{"thought": "<one short wrap-up>",
    "final_answer": "<answer in 1-3 sentences>",
    "citations": ["<wiki title 1>", "<wiki title 2>"]}}

NO markdown fences. NO prose around the JSON. NO trailing keys.

# Error handling — CRITICAL — read the result of every tool call

  • search_wikipedia → results=[]            → retry with a reworded query (drop adjectives, broaden, or pick a different angle)
  • fetch_wikipedia_summary → ok=False       → try the SECOND-best title from the previous search_wikipedia
  • verify_claim → supports="no"             → re-fetch a different summary OR rewrite the claim more conservatively
  • verify_claim → supports="partial"        → fetch more characters (larger max_chars) OR fetch a more specific page
  • calculate → ok=False                     → re-issue with a corrected expression (digits and + - * / ( ) only)
  • NEVER emit final_answer without at least one verify_claim ok=True AND supports="yes"
  • NEVER re-fetch a title you have already fetched this session — it is already in your history

# Self-check before final_answer

  Before emitting final_answer, mentally confirm:
    – Every fact in my final_answer came from a fetched Wikipedia summary IN THIS SESSION? (must be yes)
    – I called verify_claim on the keystone fact and got supports="yes"? (must be yes)
    – My citations array contains real Wikipedia titles I actually fetched? (must be yes)
  If any answer is "no", do NOT emit final_answer. Do another lookup instead.

# Conversation memory

  Your history below contains every prior tool call and result. Reuse it.
    – Do not repeat an identical tool call with identical args.
    – Do not re-fetch a title you have already seen.
    – Quote earlier facts instead of re-searching.

# Rules

  • "thought" is required on every turn.
  • Reply with EXACTLY ONE JSON object and NOTHING else. No markdown fences. No prose around it.
  • You have at most {max_iter} turns. Typical chain is 8-14 calls — use them efficiently.
"""


# ---------------------------------------------------------------------------
# Build the tool-spec for the system prompt
# ---------------------------------------------------------------------------

def build_tool_spec(tools: list[Any]) -> str:
    """Render the MCP server's tool list as a numbered, human-readable spec block."""
    lines: list[str] = []
    for i, t in enumerate(tools, 1):
        desc = (t.description or "").strip()
        schema = t.inputSchema or {}
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        arg_parts: list[str] = []
        for argname, argschema in props.items():
            typ = argschema.get("type", "any")
            if typ == "array":
                inner = argschema.get("items", {}).get("type", "any")
                typ = f"list[{inner}]"
            marker = "" if argname in required else "?"
            arg_parts.append(f"{argname}{marker}: {typ}")
        sig = f"{t.name}({', '.join(arg_parts)})"
        lines.append(f"{i}. {sig}\n   {desc}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# JSON parser — strip fences, extract first balanced object
# ---------------------------------------------------------------------------

def _extract_first_json_object(text: str) -> str:
    """Return the substring from the first `{` to its matching `}`. Tolerates trailing junk."""
    start = text.find("{")
    if start < 0:
        raise ValueError("No '{' in response.")
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
    raise ValueError("Unbalanced braces.")


def parse_agent_response(raw: str) -> dict:
    if not isinstance(raw, str):
        raise ValueError("LLM response is not a string.")
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        obj_text = _extract_first_json_object(text)
        parsed = json.loads(obj_text)

    if not isinstance(parsed, dict):
        raise ValueError("Parsed response is not an object.")

    thought = parsed.get("thought")
    if not isinstance(thought, str) or not thought.strip():
        raise ValueError('Response is missing a non-empty "thought" string.')

    has_final = isinstance(parsed.get("final_answer"), str) and parsed["final_answer"].strip()
    tool = parsed.get("tool")
    has_tool = isinstance(tool, dict) and isinstance(tool.get("name"), str)

    if has_final and has_tool:
        raise ValueError('Response has both "tool" and "final_answer" — pick one.')
    if not has_final and not has_tool:
        raise ValueError('Response must contain either "tool" or "final_answer".')

    if has_tool and not isinstance(tool.get("args"), dict):
        tool["args"] = {}

    return parsed


# ---------------------------------------------------------------------------
# History flattening
# ---------------------------------------------------------------------------

def flatten_history(history: list[dict]) -> str:
    parts: list[str] = []
    for msg in history:
        role = msg["role"]
        if role == "system":
            parts.append("[SYSTEM]\n" + msg["content"])
        elif role == "user":
            parts.append("[USER]\n" + msg["content"])
        elif role == "assistant":
            parts.append("[ASSISTANT]\n" + msg["content"])
    return "\n\n".join(parts) + "\n\n[ASSISTANT]\n"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_iter_start(iter_num: int, reasoning_type: str | None = None) -> None:
    tag = f"  [{reasoning_type}]" if reasoning_type else ""
    print(f"\n=== Iter {iter_num} ==={tag}", flush=True)


def _log_thought(thought: str) -> None:
    print(f"THOUGHT: {thought}", flush=True)


def _log_tool_call(name: str, args: dict) -> None:
    arg_str = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    if len(arg_str) > 240:
        arg_str = arg_str[:240] + "…"
    print(f"TOOL:    {name}({arg_str})", flush=True)


def _log_tool_result(name: str, result_text: str) -> None:
    display = result_text if len(result_text) <= 2000 else result_text[:2000] + " …[truncated]"
    print(f"RESULT:  {display}", flush=True)


def _log_final(answer: str, citations: list[str]) -> None:
    print(f"\n=== FINAL ANSWER ===\n{answer}", flush=True)
    if citations:
        print("\n=== CITATIONS ===", flush=True)
        for c in citations:
            print(f"  • {c}", flush=True)


# ---------------------------------------------------------------------------
# Tool-result text extraction
# ---------------------------------------------------------------------------

def _extract_text_from_tool_result(result: Any) -> str:
    try:
        chunks = result.content
    except AttributeError:
        return json.dumps(result, default=str)
    out: list[str] = []
    for c in chunks:
        t = getattr(c, "text", None)
        if t is not None:
            out.append(t)
        else:
            out.append(json.dumps(getattr(c, "__dict__", {}), default=str))
    return "\n".join(out).strip()


def _check_verify_yes(name: str, result_text: str) -> bool:
    """Did `verify_claim` return supports="yes"?"""
    if name != "verify_claim":
        return False
    try:
        parsed = json.loads(result_text)
        return (
            isinstance(parsed, dict)
            and parsed.get("ok") is True
            and parsed.get("supports") == "yes"
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def run_agent(
    session: ClientSession,
    user_topic: str,
    llm: "LLMClient",
    gemini_model_name: str,
) -> None:
    tools_resp = await session.list_tools()
    tools = tools_resp.tools

    tool_spec = build_tool_spec(tools)
    tool_names = {t.name for t in tools}

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(tool_spec=tool_spec, max_iter=MAX_ITER)

    history: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {user_topic}"},
    ]

    total_tokens = 0
    verify_yes_seen = False  # gate: ≥1 verify_claim with supports="yes" before final_answer

    for iteration in range(1, MAX_ITER + 1):
        prompt = flatten_history(history)

        try:
            raw_text, usage = await llm.generate(prompt)
        except Exception as e:
            print(f"ERROR:   LLM call failed: {e}", flush=True)
            return

        total_tokens += _total_tokens(usage)

        try:
            parsed = parse_agent_response(raw_text)
        except ValueError as e:
            _log_iter_start(iteration)
            print(f"ERROR:   Parse failed: {e}", flush=True)
            print(f"RAW:     {raw_text[:400]}", flush=True)
            history.append({"role": "assistant", "content": raw_text})
            history.append({
                "role": "user",
                "content": (
                    f"Your previous response was not valid: {e}. "
                    f"Respond again with EXACTLY ONE JSON object in the required format and NOTHING else."
                ),
            })
            continue

        reasoning_type = parsed.get("reasoning_type") if isinstance(parsed.get("reasoning_type"), str) else None
        _log_iter_start(iteration, reasoning_type)

        history.append({"role": "assistant", "content": raw_text})
        _log_thought(parsed["thought"])

        if "final_answer" in parsed and parsed["final_answer"]:
            citations = parsed.get("citations") or []
            citations_ok = (
                isinstance(citations, list)
                and len(citations) >= 1
                and all(isinstance(c, str) and c.strip() for c in citations)
            )
            if not verify_yes_seen or not citations_ok:
                reason_parts: list[str] = []
                if not verify_yes_seen:
                    reason_parts.append('no verify_claim with supports="yes" has been recorded in this session')
                if not citations_ok:
                    reason_parts.append("citations must be a non-empty list of Wikipedia titles you fetched")
                refusal = "Refusing final_answer: " + " AND ".join(reason_parts) + "."
                print(f"ERROR:   {refusal}", flush=True)
                history.append({
                    "role": "user",
                    "content": (
                        refusal + " Continue the research: call verify_claim with the keystone "
                        "fact and supporting evidence, then emit final_answer with a citations "
                        "array of real Wikipedia titles you have fetched this session."
                    ),
                })
                continue
            _log_final(parsed["final_answer"], citations)
            print(f"\n[stats] iterations={iteration}, total_tokens={total_tokens}", flush=True)
            return

        tool = parsed["tool"]
        name = tool["name"]
        args = tool.get("args", {})
        _log_tool_call(name, args)

        if name not in tool_names:
            error_text = json.dumps({
                "ok": False,
                "error": f"Unknown tool '{name}'. Available: {sorted(tool_names)}",
            })
            _log_tool_result(name, error_text)
            history.append({"role": "user", "content": f"Tool result for `{name}`:\n{error_text}"})
            continue

        try:
            result = await session.call_tool(name, args)
            result_text = _extract_text_from_tool_result(result)
        except Exception as e:
            result_text = json.dumps({"ok": False, "error": str(e)})

        if _check_verify_yes(name, result_text):
            verify_yes_seen = True

        history_blob = result_text if len(result_text) <= 4000 else result_text[:4000] + " …[truncated]"
        _log_tool_result(name, result_text)
        history.append({"role": "user", "content": f"Tool result for `{name}`:\n{history_blob}"})

    print(f"\n[stats] hit max iterations ({MAX_ITER}); total_tokens={total_tokens}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> int:
    load_dotenv()
    provider = detect_provider()
    model_name = resolve_model_name(provider, default=DEFAULT_MODEL)
    llm = LLMClient(provider=provider, model_name=model_name)

    topic = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else DEFAULT_TOPIC
    print(f"[question] {topic}", flush=True)
    print(f"[model]    {model_name} ({provider})", flush=True)

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.join(os.path.dirname(__file__), "mcp_server.py")],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await run_agent(session, topic, llm, model_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
