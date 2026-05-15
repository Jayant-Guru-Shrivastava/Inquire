# Inquire — Multi-Hop Research Agent (EAG V3 Assignment 5)

A small MCP-driven agent that answers **chained Wikipedia questions** by decomposing them into sub-questions, looking each one up, verifying the keystone facts, and synthesising a cited answer. The Session-5 deliverable is the *prompt-qualification workflow* — the README walks you through it below.

> *"What is the capital of the country where the inventor of the World Wide Web was born?"*
> → search "World Wide Web" → fetch → infer "Tim Berners-Lee" → search/fetch "Tim Berners-Lee" → infer "born in London, UK" → fetch "United Kingdom" → verify → **London** (cited).

---

## YouTube demo

> *To be added after recording.*

---

## How it works

```
                  ┌─────────────────────────────────┐
   CLI question → │   talk2mcp.py  (agent client)   │
                  │   - Qualified V2 system prompt  │
                  │   - JSON tool-call loop         │
                  │   - final_ok gate               │
                  └────────────┬────────────────────┘
                          stdio │
                  ┌────────────▼────────────────────┐
                  │   mcp_server.py  (FastMCP)      │
                  │   5 tools, Pydantic-validated   │
                  └────────────┬────────────────────┘
                               │
              ┌────────────────┼──────────────────┐
              ▼                ▼                  ▼
       wiki.py             calculate         verify_claim
   (httpx → Wikipedia)   (safe arith)     (keyword-grounded)
```

The agent never sees a `try/except` — every tool returns `{"ok": bool, "error": "..."}` and the qualified prompt teaches it to read the error and retry. Same error-visibility property as Assignment 4.

---

## The prompt qualification workflow

This is the **core Session-5 deliverable**. The lecture says:

> *"Use ChatGPT or Cursor/Claude or something that qualifies your prompt with the rules mentioned in [this](prompt_qualifier.md) prompt."*

That linked prompt is a **Prompt Evaluation Assistant** — it scores a candidate system prompt across 9 criteria and returns a JSON scorecard. You **iterate on your draft until every boolean comes back `true`**. The final iteration is the production prompt.

The repo includes [`evaluate_prompt.py`](evaluate_prompt.py) so the same evaluation can be run from the terminal (it sends the Evaluator + the target prompt to OpenRouter and prints the scorecard). Either path — ChatGPT or `evaluate_prompt.py` — satisfies the assignment; the terminal path is what's recorded in the video.

### Step 1 — Draft prompt (V1) — deliberately weak

```text
You are Inquire, a research assistant that answers questions by chaining Wikipedia lookups.

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
```

This is the prompt I'd write before reading Session 5. Lives as the `V1_PROMPT` constant in [evaluate_prompt.py](evaluate_prompt.py) for reproducibility.

### Step 2 — Evaluate V1

```bash
$ uv run python evaluate_prompt.py v1
```

Actual scorecard (gemini-3-flash-preview, May 2026):

```json
{
  "explicit_reasoning": true,
  "structured_output": true,
  "tool_separation": true,
  "conversation_loop": true,
  "instructional_framing": true,
  "internal_self_checks": false,
  "reasoning_type_awareness": true,
  "fallbacks": false,
  "overall_clarity": "The prompt clearly defines a persona and a structured multi-step process for research and tool usage."
}
```

**Two failures:** `internal_self_checks` and `fallbacks`. V1 never tells the agent to sanity-check its conclusions, and it never describes what to do when a search returns nothing or a verification fails.

### Step 3 — Iterate to fix the failing criteria

Each failure becomes an explicit section in V2:

| Failing criterion in V1 | What V2 adds |
|---|---|
| `internal_self_checks` | A "Self-check before final_answer" block plus the runtime-enforced rule that `verify_claim` must return `supports="yes"` before a final answer is accepted. |
| `fallbacks` | An "Error handling" table mapping each failure mode (`search_wikipedia` empty results, `fetch_wikipedia_summary` 404, `verify_claim` partial/no, `calculate` malformed expression) to an explicit retry strategy. |

Plus tightenings that strengthen the criteria that were already passing (tool-separation made mandatory before every external lookup; conversation-memory rules added; instructional framing hardened to "EXACTLY ONE of these two shapes — nothing else"; citations array added to the final-answer schema).

### Step 4 — Re-evaluate V2

```bash
$ uv run python evaluate_prompt.py v2
```

Actual scorecard:

```json
{
  "explicit_reasoning": true,
  "structured_output": true,
  "tool_separation": true,
  "conversation_loop": true,
  "instructional_framing": true,
  "internal_self_checks": true,
  "reasoning_type_awareness": true,
  "fallbacks": true,
  "overall_clarity": "The prompt provides a highly structured, rigorous framework for a multi-hop research agent with clear error handling and mandatory logical sequences."
}
```

**All 8 booleans are `true`.** V2 is the production prompt.

### Step 5 — Qualified prompt (V2) — what ships in `talk2mcp.py`

```text
You are Inquire, a multi-hop research agent. You answer chained questions by decomposing
them into sub-questions, looking each one up on Wikipedia, verifying the keystone facts,
then synthesising a cited answer.

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
  {"thought": "<one short sentence: why this tool now>",
   "reasoning_type": "Decomposition|Lookup|Inference|Verification|Synthesis|Computation",
   "tool": {"name": "<tool_name>", "args": {...}}}

  B) Final answer (terminal — emit ONLY after the self-check below passes):
  {"thought": "<one short wrap-up>",
   "final_answer": "<answer in 1-3 sentences>",
   "citations": ["<wiki title 1>", "<wiki title 2>"]}

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
  • Reply with EXACTLY ONE JSON object and NOTHING else.
  • You have at most {max_iter} turns. Typical chain is 8-14 calls.
```

(Lives verbatim in [talk2mcp.py](talk2mcp.py) as `SYSTEM_PROMPT_TEMPLATE`.)

The runtime also *enforces* the two criteria that needed iteration: the `final_ok` gate in `talk2mcp.py` refuses any `final_answer` that arrives before a `verify_claim` returned `supports="yes"` **and** that lacks a non-empty `citations` array. So even if the LLM ignores its own self-check, the agent loop catches it.

---

## Tools

Every tool returns `{"ok": bool, ...}` with the real error text on failure. Inputs are validated via [`models.py`](models.py) (Pydantic v2).

| # | Tool | Args | Returns | Purpose |
|---|---|---|---|---|
| 1 | `show_reasoning` | `step:int, reasoning_type:str, text:str` | `{ok, logged, step, reasoning_type}` | Tool-separation rule: the agent emits one of these every reasoning turn so the trace is auditable. No I/O. |
| 2 | `search_wikipedia` | `query:str, limit:int=5` | `{ok, query, results:[{title,snippet,page_id}], total}` | Wraps `api.php?action=query&list=search`. Empty result list is **not** an error. |
| 3 | `fetch_wikipedia_summary` | `title:str, max_chars:int=2000` | `{ok, title, page_url, extract, truncated, full_chars}` | Wraps `/api/rest_v1/page/summary/<title>`. 404 → `{ok:False, error}`. |
| 4 | `calculate` | `expression:str` | `{ok, expression, result:float}` | Safe arithmetic — regex-filtered allowed-chars then sandboxed `eval`. |
| 5 | `verify_claim` | `claim:str, evidence:str` | `{ok, supports:"yes"\|"partial"\|"no", tokens_total, tokens_matched, match_ratio, reason}` | Deterministic keyword-grounded check. ≥ 80% significant tokens overlap → "yes". |

---

## Install & run

```bash
cd Assignment-5/researcher
uv sync
cp .env.example .env   # fill in OPENROUTER_API_KEY
```

Run a single research question:

```bash
uv run python talk2mcp.py "What is the capital of the country where the inventor of the World Wide Web was born?" 2>&1 | tee run.log
```

Or open the MCP Inspector and call tools by hand:

```bash
uv run mcp dev mcp_server.py
```

---

## Demo questions used for the YouTube video

1. **"What is the capital of the country where the inventor of the World Wide Web was born?"**
   *Expected chain:* `World Wide Web` → infer "Tim Berners-Lee" → `Tim Berners-Lee` → infer "born in London, UK" → `United Kingdom` → verify → `London`.

2. **"Who painted the ceiling of the chapel where the modern Pope is elected?"**
   *Expected chain:* `papal conclave` → infer "Sistine Chapel" → `Sistine Chapel` → infer "ceiling by Michelangelo" → verify → `Michelangelo`.

3. **"What is the difference in population between Oslo and Stockholm?"**
   *Demonstrates `calculate`.* Fetch Oslo, extract population → fetch Stockholm, extract → `calculate(pop_stockholm - pop_oslo)` → cited answer.

---

## Sample iteration log (illustrative)

```
[question] What is the capital of the country where the inventor of the World Wide Web was born?
[model]    google/gemini-3-flash-preview (openrouter)

=== Iter 1 ===  [Decomposition]
THOUGHT: I need three facts: WWW inventor, their birthplace, that country's capital.
TOOL:    show_reasoning(step=1, reasoning_type="Decomposition", text="(1) Who invented the WWW? (2) Where was that person born? (3) What is that country's capital?")
RESULT:  {"ok":true,"logged":true,"step":1,"reasoning_type":"Decomposition"}

=== Iter 2 ===  [Lookup]
THOUGHT: First sub-question — who invented the WWW.
TOOL:    search_wikipedia(query="World Wide Web inventor")
RESULT:  {"ok":true,"results":[{"title":"World Wide Web","snippet":"...","page_id":"33139"}, ...]}
...
=== Iter 8 ===  [Verification]
TOOL:    verify_claim(claim="Tim Berners-Lee was born in London", evidence="...")
RESULT:  {"ok":true,"supports":"yes","tokens_matched":3,"tokens_total":3,"match_ratio":1.0,...}

=== Iter 9 ===  [Synthesis]
TOOL:    show_reasoning(reasoning_type="Synthesis", text="WWW inventor = Berners-Lee; born London; UK capital = London.")
RESULT:  {"ok":true,...}

=== FINAL ANSWER ===
London. Tim Berners-Lee invented the World Wide Web; he was born in London,
which is the capital of the United Kingdom.

=== CITATIONS ===
  • World Wide Web
  • Tim Berners-Lee
  • United Kingdom

[stats] iterations=9, total_tokens=...
```

A real `run.log` from the YouTube demo recording will be appended to this README after the video is shot.

---

## Files

| File | Purpose |
|---|---|
| [talk2mcp.py](talk2mcp.py) | Agent client. JSON tool-call loop, final_ok gate, V2 prompt. |
| [mcp_server.py](mcp_server.py) | FastMCP server with the five tools. |
| [models.py](models.py) | Pydantic v2 input models — one source of truth per tool. |
| [wiki.py](wiki.py) | Async Wikipedia client (httpx). |
| [llm.py](llm.py) | Provider-agnostic LLM client (OpenRouter / Vertex / AI Studio). |
| [prompt_qualifier.md](prompt_qualifier.md) | The Session-5 Prompt Evaluation Assistant. |
| [evaluate_prompt.py](evaluate_prompt.py) | Runs the Evaluator over V1 / V2 from the terminal. |
| [pyproject.toml](pyproject.toml) | `uv` project manifest. |

---

## Reuse from Assignment 4

`llm.py` is copied verbatim from `Assignment-4/pr-drafter/llm.py`. The JSON parser (`_extract_first_json_object`, `parse_agent_response`), the MCP stdio spawn, the iteration logging, the rate-limited `generate` and the gate pattern (renamed `write_ok` → `final_ok` with different trigger conditions) all come from `Assignment-4/pr-drafter/talk2mcp.py`. The httpx pattern in `wiki.py` was adapted from `Assignment-4/talk2mcp/images.py`.

The genuinely new code in this assignment: the Pydantic models in `models.py`, the five Wikipedia/verify tools in `mcp_server.py`, the V2 system prompt, and the prompt-qualification workflow itself (the central Session-5 deliverable).
