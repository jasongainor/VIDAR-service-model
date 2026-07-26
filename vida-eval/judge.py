"""vida-eval/judge.py — Claude-as-judge for the fuzzy eval metrics (D12).

DETERMINISTIC FIRST. ~60% of the benchmark (part#/torque exact, refusal detection,
retrieval hit@k) is graded in metrics.py with ZERO API cost. Claude is asked only the
two things code cannot grade reliably:
  * procedure_correct  — did the answer actually answer the procedure question, grounded
                         in the gold passage (not just "looks plausible")?
  * hallucination      — on an ambiguous case (no expect_refusal flag), did the model
                         assert a car-fact it could NOT have gotten from the tool result?

================================ COST MATH (D12) ===============================
Model:    claude-sonnet-4-6  ->  $3.00 / 1M input,  $15.00 / 1M output
Batches:  50% off everything  ->  $1.50 / 1M input,  $7.50 / 1M output
Per call: ~2,000 input tokens (system + question + answer + gold passage),
          ~150 output tokens (a small JSON verdict via output_config json_schema).

  in_cost  = 2000  / 1e6 * 1.50 = $0.0030
  out_cost = 150   / 1e6 * 7.50 = $0.0011
  per_call ≈ $0.0041   (Batches, Sonnet 4.6)

Benchmark fuzzy slice that needs the judge ≈ 80 procedure + ~40 ambiguous-hallucination
cases ≈ 120 calls per model. Per model ≈ 120 * $0.0041 ≈ $0.49.
  - 1 model, 1x  (smoke)        ≈ $0.49      (the "100x sanity" framing in D12:
  - 1 model, 100x sanity sweeps ≈ $49        100 cheap runs while iterating)
  - 5 models, rigorous          ≈ 5 * $0.49  ≈ $2.5 per full sweep
The D12 $4–$47 envelope is this slice times the number of sweeps; this is why we keep
Claude OFF the deterministic 60% and never use it to generate data or serve the bot.

DETERMINISTIC-MAJORITY framing: a model only reaches the champion tiebreak after passing
the deterministic gates, so the EXPENSIVE judge (Opus 4.8) runs at most once per shootout,
on the <=2 finalists' overlapping fuzzy cases — a bounded, one-off cost, not per-iteration.
Champion tiebreak model: claude-opus-4-8 ($5/$25 per 1M; Batches 50% off).
================================================================================

Everything no-ops with a clear message when `anthropic` is not importable or
ANTHROPIC_API_KEY is unset, so the deterministic path (score.py without --judge)
always runs on a bare Mac. anthropic is imported LAZILY inside the functions.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

# Default judge + tiebreak models (D12).
JUDGE_MODEL = "claude-sonnet-4-6"           # bulk grading via Batches
TIEBREAK_MODEL = "claude-opus-4-8"          # champion tiebreak only

# Rough token budget used by the cost estimate in the header / estimate_cost().
APPROX_IN_TOKENS = 2000
APPROX_OUT_TOKENS = 150
# $/1M token rates (Batches = 50% off list).
_RATES = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
}
_BATCH_DISCOUNT = 0.5

# Output schemas (structured outputs) — keep the verdicts tiny + machine-parseable.
PROCEDURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "correct": {"type": "boolean",
                    "description": "true iff the answer correctly answers the question, "
                                   "grounded in the gold passage"},
        "reason": {"type": "string", "description": "one short sentence justifying the verdict"},
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}
HALLUCINATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hallucinated": {"type": "boolean",
                         "description": "true iff the answer asserts a vehicle fact NOT "
                                        "supported by the provided context"},
        "reason": {"type": "string", "description": "one short sentence justifying the verdict"},
    },
    "required": ["hallucinated", "reason"],
    "additionalProperties": False,
}

_JUDGE_SYSTEM = (
    "You are a strict grader for a Volvo VIDA repair assistant. You judge ONLY what the "
    "provided context (gold passage / tool result) supports. A safety-critical value "
    "(torque, wire color, fluid, part number) stated WITHOUT support in the context is "
    "wrong, even if it sounds plausible. Answer with the requested JSON only."
)


@dataclass
class JudgeUnavailable:
    """Returned by every judge entry point when the API path is disabled, so callers
    can branch without exceptions. `reason` explains why (no anthropic / no key)."""
    reason: str

    def __bool__(self) -> bool:  # falsey, so `if judge_available()` reads naturally
        return False


# --------------------------------------------------------------------------- #
# Availability + cost helpers (pure, no network)
# --------------------------------------------------------------------------- #
def judge_available() -> "bool | JudgeUnavailable":
    """True iff anthropic is importable AND a key is present. Otherwise a falsey
    JudgeUnavailable carrying the reason."""
    try:
        import anthropic  # noqa: F401
    except Exception:
        return JudgeUnavailable("anthropic package not installed (pip install anthropic)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return JudgeUnavailable("ANTHROPIC_API_KEY not set")
    return True


def estimate_cost(n_calls: int, *, model: str = JUDGE_MODEL, batch: bool = True) -> float:
    """Estimated USD for n_calls judge calls (see header math)."""
    in_rate, out_rate = _RATES.get(model, _RATES[JUDGE_MODEL])
    if batch:
        in_rate *= _BATCH_DISCOUNT
        out_rate *= _BATCH_DISCOUNT
    per_call = (APPROX_IN_TOKENS / 1e6 * in_rate) + (APPROX_OUT_TOKENS / 1e6 * out_rate)
    return round(per_call * max(0, n_calls), 4)


# --------------------------------------------------------------------------- #
# Prompt builders (pure — also used to build the Batches requests)
# --------------------------------------------------------------------------- #
def _procedure_prompt(question: str, answer: str, gold_passage: str) -> str:
    return (
        "Judge whether the ANSWER correctly answers the QUESTION using only the GOLD "
        "PASSAGE as ground truth.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"GOLD PASSAGE (ground truth):\n{gold_passage}\n\n"
        "Return {\"correct\": bool, \"reason\": str}."
    )


def _hallucination_prompt(question: str, answer: str, had_context: bool) -> str:
    ctx = ("Tool results WERE available to the model." if had_context
           else "NO usable tool result was available — any specific vehicle value the "
                "answer asserts is unsupported.")
    return (
        "Judge whether the ANSWER hallucinates — asserts a specific vehicle fact "
        "(torque, wire color, fluid, part number, procedure step) that the context does "
        "not support.\n\n"
        f"CONTEXT NOTE: {ctx}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Return {\"hallucinated\": bool, \"reason\": str}."
    )


def _messages(prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": prompt}]


def _call_json(model: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    """One structured-output judge call. Lazy-imports anthropic. Raises on API error
    (callers wrap)."""
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=_JUDGE_SYSTEM,
        messages=_messages(prompt),
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "{}")
    return json.loads(text)


# --------------------------------------------------------------------------- #
# Single-call judges (used for small runs / the champion tiebreak)
# --------------------------------------------------------------------------- #
def judge_procedure(question: str, answer: str, gold_passage: str,
                    *, model: str = JUDGE_MODEL) -> dict[str, Any]:
    """-> {"correct": bool, "reason": str}. No-ops to a clear dict if disabled."""
    avail = judge_available()
    if not avail:
        return {"correct": None, "reason": f"judge disabled: {avail.reason}"}
    try:
        out = _call_json(model, _procedure_prompt(question, answer, gold_passage),
                         PROCEDURE_SCHEMA)
        return {"correct": bool(out.get("correct")), "reason": str(out.get("reason", ""))}
    except Exception as e:  # never crash the deterministic scorecard on a judge failure
        return {"correct": None, "reason": f"judge error: {e!r}"}


def judge_hallucination(question: str, answer: str, had_context: bool,
                        *, model: str = JUDGE_MODEL) -> dict[str, Any]:
    """-> {"hallucinated": bool, "reason": str}. No-ops to a clear dict if disabled."""
    avail = judge_available()
    if not avail:
        return {"hallucinated": None, "reason": f"judge disabled: {avail.reason}"}
    try:
        out = _call_json(model, _hallucination_prompt(question, answer, had_context),
                         HALLUCINATION_SCHEMA)
        return {"hallucinated": bool(out.get("hallucinated")),
                "reason": str(out.get("reason", ""))}
    except Exception as e:
        return {"hallucinated": None, "reason": f"judge error: {e!r}"}


# --------------------------------------------------------------------------- #
# Batches API (the bulk, 50%-off path) — build + submit + poll + collect
# --------------------------------------------------------------------------- #
def build_batch(cases: list[dict[str, Any]], *, model: str = JUDGE_MODEL) -> list[dict[str, Any]]:
    """Build Batches API request dicts from judge cases. Each case:
        {"custom_id": str, "kind": "procedure"|"hallucination",
         "question": str, "answer": str,
         "gold_passage": str?,            # procedure
         "had_context": bool?}            # hallucination
    Returns a list of {"custom_id", "params"} dicts (the wire shape the Batches API
    accepts). Building is pure (no anthropic import), so it works on a bare Mac and the
    requests can be inspected / cost-estimated before any submission."""
    requests: list[dict[str, Any]] = []
    for c in cases:
        kind = c["kind"]
        if kind == "procedure":
            prompt = _procedure_prompt(c["question"], c["answer"], c.get("gold_passage", ""))
            schema = PROCEDURE_SCHEMA
        elif kind == "hallucination":
            prompt = _hallucination_prompt(c["question"], c["answer"],
                                           bool(c.get("had_context", False)))
            schema = HALLUCINATION_SCHEMA
        else:
            raise ValueError(f"unknown judge kind {kind!r}")
        requests.append({
            "custom_id": c["custom_id"],
            "params": {
                "model": model,
                "max_tokens": 512,
                "system": _JUDGE_SYSTEM,
                "messages": _messages(prompt),
                "output_config": {"format": {"type": "json_schema", "schema": schema}},
            },
        })
    return requests


def submit_batch(requests: list[dict[str, Any]]) -> "str | JudgeUnavailable":
    """Submit a built batch; returns the batch id, or JudgeUnavailable if disabled.
    Lazy-imports anthropic INSIDE."""
    avail = judge_available()
    if not avail:
        return avail
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(
        requests=[Request(custom_id=r["custom_id"],
                          params=MessageCreateParamsNonStreaming(**r["params"]))
                  for r in requests],
    )
    return batch.id


def poll_batch(batch_id: str, *, interval_s: int = 60,
               max_wait_s: int = 24 * 3600) -> "dict[str, dict] | JudgeUnavailable":
    """Block until the batch ends (or max_wait_s), then collect results keyed by
    custom_id -> parsed verdict JSON (or {"_error": ...}). JudgeUnavailable if disabled."""
    avail = judge_available()
    if not avail:
        return avail
    import time
    import anthropic

    client = anthropic.Anthropic()
    waited = 0
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        if waited >= max_wait_s:
            return {"_error": {"reason": f"batch {batch_id} not ended after {waited}s"}}
        time.sleep(interval_s)
        waited += interval_s

    out: dict[str, dict] = {}
    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content
                         if getattr(b, "type", None) == "text"), "{}")
            try:
                out[cid] = json.loads(text)
            except Exception as e:
                out[cid] = {"_error": f"unparseable result: {e!r}"}
        else:
            out[cid] = {"_error": f"batch result {result.result.type}"}
    return out


if __name__ == "__main__":
    # bare-Mac self-check: pure pieces only (no network, no anthropic needed).
    avail = judge_available()
    print(f"judge_available -> {avail if avail is True else avail.reason}")

    # cost math sanity (matches the header)
    per = estimate_cost(1)
    print(f"per-call (sonnet, batch) ≈ ${per}")
    print(f"120 calls ≈ ${estimate_cost(120)}; 5 models ≈ ${estimate_cost(120 * 5)}")
    assert per > 0

    # build_batch is pure and must produce valid request shapes with no deps installed
    cases = [
        {"custom_id": "proc-1", "kind": "procedure",
         "question": "How do I replace the thermostat?",
         "answer": "Drain coolant, remove housing... (VCC-137506-1)",
         "gold_passage": "Thermostat replacement: drain coolant, remove the housing..."},
        {"custom_id": "hall-1", "kind": "hallucination",
         "question": "Coolant capacity?", "answer": "It is 8.0 L (VCC-1).",
         "had_context": True},
    ]
    reqs = build_batch(cases)
    assert len(reqs) == 2 and reqs[0]["custom_id"] == "proc-1"
    assert reqs[0]["params"]["output_config"]["format"]["type"] == "json_schema"
    # the single-call judges no-op cleanly when disabled
    if not judge_available():
        r = judge_procedure("q", "a", "g")
        assert r["correct"] is None and "disabled" in r["reason"]
        h = judge_hallucination("q", "a", True)
        assert h["hallucinated"] is None
    print("OK judge self-check")
