"""vida-eval/score.py — grade a model's answers over the frozen benchmark (Part 10).

DETERMINISTIC FIRST (D12): part#/torque exact, refusal detection, retrieval hit@k cost
$0 (metrics.py). Only procedure-correctness (and ambiguous hallucination) escalate to
the Claude judge (judge.py), and only when --judge is passed and the key is present.

Public entry point used by the eval executor:
    score_run(cases, answers, *, model, config, dataset_version, router_base_url) -> scorecard
where `cases` are frozen-benchmark records (benchmark_gen.py) and `answers` are
[{"id", "answer", "tool_refs"}] produced by replaying the candidate model.

CLI:
    python vida-eval/score.py --benchmark vida-eval/benchmark.jsonl \
        --answers run/answers.jsonl --model qwen3-4b --config r16 [--judge] [--post]

Loaded by path from eval_bench.py (hyphen dir); also runnable standalone. Stdlib +
(lazy) judge/anthropic. Bootstraps sys.path so siblings + grounding import cleanly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _d in (_HERE, _REPO_ROOT):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import metrics  # noqa: E402  (sibling vida-eval/metrics.py)
import judge     # noqa: E402  (sibling vida-eval/judge.py)


# --------------------------------------------------------------------------- #
# ref helpers
# --------------------------------------------------------------------------- #
def _ref_hit(gold_ref: Optional[str], tool_refs: list[str]) -> Optional[bool]:
    """True iff the gold source id was among the refs the tools served. None when
    there's no gold ref to check (so it doesn't count toward hit@k)."""
    if not gold_ref:
        return None
    # EQUALITY only — both the gold ref and every served ref come from the same
    # `vida_doc_ref or doc_id` rule, so a substring test ('VCC-1' in 'VCC-1234-9')
    # would falsely count a prefix collision as a hit and inflate hit@k.
    g = gold_ref.strip().lower()
    return any(g == (r or "").strip().lower() for r in tool_refs)


def _user_question(case: dict) -> str:
    for m in case.get("messages", []):
        if m.get("role") == "user":
            # strip a leading [VEHICLE]...[/VEHICLE] block
            c = m.get("content", "")
            return c.split("[/VEHICLE]\n", 1)[-1] if "[/VEHICLE]" in c else c
    return case.get("question", "")


# --------------------------------------------------------------------------- #
# Per-case grading (deterministic; judge only where flagged)
# --------------------------------------------------------------------------- #
def grade_case(case: dict, ans: dict, *, use_judge: bool = False) -> dict:
    md = case.get("metadata", case)
    atype = md.get("answer_type")
    text = ans.get("answer", "") or ""
    refs = ans.get("tool_refs", []) or []
    pc: dict[str, Any] = {}

    if atype == "torque":
        pc["torque_exact"] = metrics.torque_exact(text, md.get("gold_value"))
        hit = _ref_hit(md.get("gold_doc_ref"), refs)
        if hit is not None:
            pc["retrieval_hit"] = hit

    elif atype == "part_number":
        pc["part_number_exact"] = metrics.part_number_exact(text, md.get("gold_part", ""))

    elif atype == "procedure":
        hit = _ref_hit(md.get("gold_doc_ref"), refs)
        if hit is not None:
            pc["retrieval_hit"] = hit
        if use_judge and judge.judge_available() is True:
            v = judge.judge_procedure(_user_question(case), text, md.get("gold_passage", ""))
            if v.get("correct") is not None:
                pc["procedure_correct"] = bool(v["correct"])

    elif atype == "pinout":
        gold = (md.get("gold_text") or "").lower()
        pc["procedure_correct"] = bool(gold and gold[:24] in text.lower())

    elif atype == "figure":
        gold_ref = (md.get("gold_doc_ref") or "")
        pc["figure_correct"] = bool(
            _ref_hit(gold_ref, refs) or (gold_ref and gold_ref.lower() in text.lower()))

    elif atype == "refusal":
        # A refusal only counts if the model ALSO did not quote a value. An answer that
        # asks for the VIN *and* asserts "15 Nm" is a hallucination, not a refusal.
        refused = metrics.detect_refusal(text) and not metrics.asserts_value(text)
        if md.get("expect_refusal"):
            pc["refusal_correct"] = refused
            pc["hallucinated"] = not refused   # answered when it should have refused

    elif atype == "cross_vehicle":
        if md.get("gold_part"):
            pc["part_number_exact"] = metrics.part_number_exact(text, md["gold_part"])
        # must flag verify, must NOT assert an exact fit
        pc["procedure_correct"] = "verify before use" in text.lower()

    return pc


def score_run(cases: list[dict], answers: list[dict], *, model: str = "unknown",
              config: str = "", dataset_version: str = "v1",
              router_base_url: str = "", use_judge: bool = False) -> dict[str, Any]:
    """Grade answers against cases -> scorecard (metrics.aggregate shape). `router_base_url`
    is accepted for signature parity with the eval executor (judge/gen route through it /
    Claude); scoring itself is local. Unmatched ids are skipped (logged in the scorecard)."""
    by_id = {a.get("id"): a for a in answers}
    per_case: list[dict] = []
    missing = 0
    for case in cases:
        cid = case.get("id")
        ans = by_id.get(cid)
        if ans is None:
            missing += 1
            continue
        per_case.append(grade_case(case, ans, use_judge=use_judge))
    sc = metrics.aggregate(per_case, model=model, config=config,
                           dataset_version=dataset_version)
    sc["answered"] = len(per_case)
    sc["missing_answers"] = missing
    return sc


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="score a model's answers over the frozen benchmark")
    ap.add_argument("--benchmark", default="vida-eval/benchmark.jsonl")
    ap.add_argument("--answers", required=True, help="jsonl of {id, answer, tool_refs}")
    ap.add_argument("--model", default="unknown")
    ap.add_argument("--config", default="")
    ap.add_argument("--dataset-version", default="v1")
    ap.add_argument("--judge", action="store_true", help="enable the Claude judge for procedure cases")
    args = ap.parse_args(argv)

    cases = _read_jsonl(args.benchmark)
    answers = _read_jsonl(args.answers)
    sc = score_run(cases, answers, model=args.model, config=args.config,
                   dataset_version=args.dataset_version, use_judge=args.judge)

    m = sc["metrics"]
    def r(name):
        v = m.get(name)
        return v if isinstance(v, (int, float)) else (v or {}).get("rate")
    print(f"{args.model}/{args.config} n={sc['n']} answered={sc['answered']} "
          f"missing={sc['missing_answers']}")
    print(f"  part#={r('part_number_exact')} torque={r('torque_exact')} "
          f"refusal={r('refusal_correct')} halluc={m.get('hallucination_rate')} "
          f"hit@k={r('retrieval_hit_at_k')} proc={r('procedure_correct')} fig={r('figure_correct')}")
    print(f"  PASSED={sc['passed']}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # bare-Mac self-check: synthetic cases + answers, deterministic graders only.
        cases = [
            {"id": "torque-0", "metadata": {"answer_type": "torque", "gold_value": 15,
             "gold_doc_ref": "VCC-112365-1"}},
            {"id": "part-0", "metadata": {"answer_type": "part_number", "gold_part": "1271998"}},
            {"id": "refusal-0", "metadata": {"answer_type": "refusal", "expect_refusal": True}},
            {"id": "figure-0", "metadata": {"answer_type": "figure", "gold_doc_ref": "VCC-9-1"}},
        ]
        answers = [
            {"id": "torque-0", "answer": "15 Nm (VCC-112365-1).", "tool_refs": ["VCC-112365-1"]},
            {"id": "part-0", "answer": "1271998 is the heated oxygen sensor.", "tool_refs": []},
            {"id": "refusal-0", "answer": "I couldn't find a verified value. Give me the VIN.", "tool_refs": []},
            {"id": "figure-0", "answer": "Here is the figure (VCC-9-1).", "tool_refs": ["VCC-9-1"]},
        ]
        sc = score_run(cases, answers, model="self", config="check")
        m = sc["metrics"]
        assert m["torque_exact"]["rate"] == 1.0, m
        assert m["part_number_exact"]["rate"] == 1.0, m
        assert m["refusal_correct"]["rate"] == 1.0 and m["hallucination_rate"] == 0.0, m
        assert m["figure_correct"]["rate"] == 1.0, m
        assert m["retrieval_hit_at_k"]["rate"] == 1.0, m
        # a hallucinating refusal flips the gate
        bad = score_run(
            [{"id": "r", "metadata": {"answer_type": "refusal", "expect_refusal": True}}],
            [{"id": "r", "answer": "Tire pressure is 42 psi.", "tool_refs": []}], model="self")
        assert bad["metrics"]["hallucination_rate"] == 1.0, bad["metrics"]
        # an answer that ELICITS but also asserts a torque is a hallucination, not a refusal
        tricky = score_run(
            [{"id": "t", "metadata": {"answer_type": "refusal", "expect_refusal": True}}],
            [{"id": "t", "answer": "It's 15 Nm, but give me the VIN, engine code, or model and year to confirm.",
              "tool_refs": []}], model="self")
        assert tricky["metrics"]["hallucination_rate"] == 1.0, tricky["metrics"]
        # equality-only ref hit: a prefix collision must NOT count as a hit
        assert _ref_hit("VCC-1", ["VCC-1234-9"]) is False
        assert _ref_hit("VCC-1234-9", ["VCC-1234-9"]) is True
        print("score.py self-check OK")
    else:
        raise SystemExit(main())
