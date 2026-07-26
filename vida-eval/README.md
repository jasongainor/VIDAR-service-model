# vida-eval — regression set for the VIDA S60R assistant

Purpose: catch retrieval/prompt regressions — especially the two-turn
contamination bug ("follow-up question gets answered as if it were the
previous topic") — so they never silently return.

## What's in here

- `cases.json` — 12 cases: 8 grounded-fact questions (torques, capacities,
  pinouts — all ground-truthed against the store during phase-1 acceptance),
  2 abstention cases (spec missing from VIDA 2014D; out-of-corpus vehicle),
  and 2 **two-turn contamination cases** (unrelated A→B and adjacent A→B).
  Grading is mechanical, case-insensitive substring:
  - `must_mention` — all must appear in that turn's final answer
  - `must_not_say` — none may appear
  - `tool_must_not_contain` — none may appear in any tool-call argument that
    turn (catches contamination at the retrieval layer even when the visible
    answer looks clean)
- `run.py` — the runner. Multi-turn history mirrors the Open WebUI chat:
  follow-up turns see only the final assistant text, never tool dumps.
- `results/` — JSON outputs (gitignored).

## Running it

Prereqs: LM Studio serving on :1234 with `qwen/qwen3.6-35b-a3b` available,
and the vida-kb MCP server on :8765 (`docs/handoff.md` → Runbook).

```sh
cd ~/Documents/vida-bot
uv run python vida-eval/run.py                                  # full set (~10-20 min)
uv run python vida-eval/run.py --case contamination-unrelated   # just the A→B check
uv run python vida-eval/run.py --json-out vida-eval/results/run.json
uv run python vida-eval/run.py --system-file /tmp/candidate-prompt.md  # try a prompt edit
```

Exit code 0 = all pass. Non-zero = read the per-case FAIL lines.

**Run this after ANY of:** a system-prompt edit (before syncing it into the
webui.db presets), a `src/server.py` retrieval change, a store rebuild, an
Open WebUI settings/version change, or swapping the LM Studio model.

## Backends

- `--backend native` (default): drives LM Studio + the MCP server directly —
  the same loop Open WebUI runs server-side for native function-calling
  presets. No Open WebUI credentials needed; tool calls are fully observable.
- `--backend owui`: goes through Open WebUI's OpenAI-compatible API
  (`$OWUI_URL`, default `http://127.0.0.1:8080`; requires `$OWUI_TOKEN`).
  Caveat: with the bare API, a **native** function-calling preset hands tool
  calls back to the API client instead of executing them server-side — so use
  this backend with a default-function-calling preset, and note tool-level
  checks are skipped (tool calls aren't visible through the API).

## Known brittleness (accepted)

Citation checks (`VCC-…` in `must_mention`) assume the model cites doc refs,
which the system prompt mandates. A model that answers correctly but cites
differently will show as FAIL — inspect the JSON before declaring regression.
Small local models are nondeterministic even at temperature 0.1; a single
flaky failure ≠ regression, rerun the case before investigating.
