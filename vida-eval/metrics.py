"""vida-eval/metrics.py — the fine-tune eval scorecard, gates, and pure graders.

This is the measurement instrument the candidate models are judged by (D16: reuse
`vida-eval/`, don't fork). It defines:

  * the scorecard JSON shape pinned in docs/BUILD_CONTRACT.md ("eval scorecard"),
  * the acceptance GATES a model must clear to "pass",
  * pure, deterministic graders (part#, torque, refusal) that cost $0 — no model,
    no network. score.py calls these for ~60% of the benchmark and only escalates
    procedure-correctness / ambiguous hallucination to judge.py (Claude API).
  * aggregate(per_case) -> scorecard, and is_champion_if(new, champ) -> bool, for
    callers that rank candidate fine-tunes.

Why these graders are deterministic: a wrong torque or part number is a safety
failure, not a style nit. We grade those exactly (string-normalized), so the
gate (>=0.995) is a real number, not an LLM's opinion. Claude only judges the
genuinely fuzzy things (did the procedure answer the question; did the model
hallucinate on an ambiguous case) — see judge.py for the cost framing.

Pure stdlib (re, json) so it imports and self-checks on a bare Mac.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Acceptance gates (docs/BUILD_CONTRACT.md "eval scorecard")
# A model PASSES only if every gated metric clears its threshold.
#   part_number_exact >= 0.995   torque_exact >= 0.995
#   hallucination_rate <= 0.005  refusal_correct >= 0.98
# retrieval_hit_at_k is gated too (it bounds the ceiling of grounded accuracy).
# --------------------------------------------------------------------------- #
GATES: dict[str, float] = {
    "part_number_exact": 0.995,   # >= floor
    "torque_exact": 0.995,        # >= floor
    "hallucination_rate": 0.005,  # <= ceiling (lower is better)
    "refusal_correct": 0.98,      # >= floor
    "retrieval_hit_at_k": 0.90,   # >= floor; bounds grounded accuracy
}

# Direction of each gate: True == higher is better (>= gate), False == lower is
# better (<= gate). Hallucination is the only "lower is better" gate.
_GATE_HIGHER_BETTER: dict[str, bool] = {
    "part_number_exact": True,
    "torque_exact": True,
    "hallucination_rate": False,
    "refusal_correct": True,
    "retrieval_hit_at_k": True,
}

# Champion weighting: hallucination is weighted highest (a model that invents
# values is disqualifying), then the exact-value gates, then refusal/retrieval.
# Used by is_champion_if to compute a single comparable acceptance score.
_CHAMPION_WEIGHTS: dict[str, float] = {
    "hallucination_rate": 3.0,    # weighted highest (penalty term)
    "part_number_exact": 2.0,
    "torque_exact": 2.0,
    "refusal_correct": 1.5,
    "retrieval_hit_at_k": 1.0,
    "procedure_correct": 1.0,
    "figure_correct": 0.5,
}

# default k for retrieval hit@k (matches the served search_docs limit=8)
DEFAULT_K = 8


# --------------------------------------------------------------------------- #
# Deterministic graders — pure functions, $0
# --------------------------------------------------------------------------- #
def _norm_part(s: str) -> str:
    """Normalize a Volvo part number for exact comparison: uppercase, drop all
    non-alphanumerics (spaces, dashes), so '30 650 014' == '30650014'."""
    return re.sub(r"[^0-9A-Za-z]", "", (s or "")).upper()


# A VCC document citation, e.g. "(VCC-112365-1)" / "VCC-118802". Stripped before
# part-number matching so a DOCUMENT reference can never satisfy the part gate.
_VCC_CITE_RE = re.compile(r"\(?\s*VCC-?[0-9A-Za-z]+-?[0-9A-Za-z]*\s*\)?", re.I)
# A part-number-shaped run: digit-led/digit-ended with optional internal spaces or
# dashes, so "30 650 014" is one candidate token but surrounding prose words are never
# glued in. (Volvo part numbers in this corpus are numeric; extend if alpha parts appear.)
_PART_TOKEN_RE = re.compile(r"\d(?:[\d \-]*\d)?")
# A bare "asserted a value" probe: 5+ consecutive digits == part-number-shaped (torque
# values are 2-3 digits and handled by _parse_torques). Used by asserts_value().
_PART_RE = re.compile(r"\b\d{5,}\b")


def part_number_exact(pred: str, gold: str) -> bool:
    """True iff a candidate part token in pred equals the gold part EXACTLY after
    normalization. VCC citation tokens are removed first, so a document reference like
    '(VCC-1234-1)' can never satisfy the part-number safety gate, and matching is
    whole-token equality (not substring) so 1234567 does not match 12345678. The model
    answers in prose ('The part number is 30650014 (VCC-…)'); the prose token matches,
    the citation does not. See docs/AUDIT-2026-06-20.md."""
    g = _norm_part(gold)
    if not g:
        return False
    cleaned = _VCC_CITE_RE.sub(" ", pred or "")
    for tok in _PART_TOKEN_RE.findall(cleaned):
        if _norm_part(tok) == g:
            return True
    return False


# torque values like "15 Nm", "15Nm", "15 N·m", "15 N-m", "15 newton metres"
_TORQUE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(?:n[\s·.\-]?m|nm|newton[\s\-]?met(?:re|er)s?)\b",
    re.I,
)


def _parse_torques(text: str) -> list[float]:
    """All torque magnitudes (in Nm) found in free text."""
    out: list[float] = []
    for m in _TORQUE_RE.finditer(text or ""):
        try:
            out.append(float(m.group(1).replace(",", ".")))
        except ValueError:
            continue
    return out


def torque_exact(pred_text: str, gold_value_nm: float, tol: float = 0.05) -> bool:
    """True iff a torque value matching gold_value_nm (within a tiny absolute
    tolerance for float/rounding noise) is quoted in pred_text. Normalizes the
    unit form ('15 Nm' / '15Nm' / '15 N·m'). tol guards float representation, NOT
    a real engineering tolerance — the spec value must be the one quoted."""
    try:
        gold = float(gold_value_nm)
    except (TypeError, ValueError):
        return False
    for v in _parse_torques(pred_text):
        if abs(v - gold) <= tol:
            return True
    return False


# A refusal/elicitation: the model declined to assert a value and asked to narrow
# the search (matches gc.REFUSAL_TEMPLATE / gc.ELICITATION_TEMPLATE shape without
# importing them, so this stays a pure local grader).
_REFUSAL_CUES = (
    r"could ?n'?t find",
    r"couldn'?t verify",
    r"could not (?:find|verify)",
    r"no verified",
    r"nothing in the verified",
    r"not able to verify",
    r"unable to verify",
    r"don'?t have a verified",
    r"i (?:can'?t|cannot) confirm",
)
# Elicitation cues are INTERROGATIVE/IMPERATIVE asks, never a bare mention. A grounded
# answer that merely says "the model and year affect the part" must NOT read as a
# refusal — so we match "what model/year", "give me the VIN", etc., not "model and
# year" on its own. The asserted-value gate in score.grade_case is the second guard.
_ELICIT_CUES = (
    r"which vehicle",
    r"give me the vin",
    r"what'?s the vin",
    r"what (?:model|year|engine)",
    r"(?:tell|give) me .{0,30}(?:vin|engine code|model|year)",
    r"need the car",
    r"so i can narrow",
    r"to narrow (?:the search|it|down|your search)",
)
_REFUSAL_RE = re.compile("|".join(f"(?:{p})" for p in _REFUSAL_CUES), re.I)
_ELICIT_RE = re.compile("|".join(f"(?:{p})" for p in _ELICIT_CUES), re.I)


def detect_refusal(text: str) -> bool:
    """True iff the answer is a refusal OR an elicitation (it declined to assert a
    value and asked for the VIN/engine/model+year). Both count as 'correctly did not
    invent a value'. NOTE: callers grading the refusal gate must ALSO check
    asserts_value() — an answer can both ask for the car AND wrongly quote a value, and
    that is a hallucination, not a refusal (see score.grade_case)."""
    t = text or ""
    return bool(_REFUSAL_RE.search(t) or _ELICIT_RE.search(t))


def asserts_value(text: str) -> bool:
    """True iff the answer quotes a concrete torque or part-number-shaped value (after
    removing VCC citation tokens, whose digits are not asserted values). Used to deny
    refusal credit to an answer that actually asserted a value."""
    cleaned = _VCC_CITE_RE.sub(" ", text or "")
    if _parse_torques(cleaned):
        return True
    return bool(_PART_RE.search(cleaned))


# --------------------------------------------------------------------------- #
# Per-metric counters -> rate
# --------------------------------------------------------------------------- #
def _rate_block(passed: int, n: int, *, k: Optional[int] = None) -> dict[str, Any]:
    block: dict[str, Any] = {"pass": passed, "n": n,
                             "rate": (passed / n) if n else 0.0}
    if k is not None:
        block["k"] = k
    return block


# --------------------------------------------------------------------------- #
# Aggregate per-case grades -> the scorecard JSON (BUILD_CONTRACT shape)
# --------------------------------------------------------------------------- #
def aggregate(per_case: list[dict[str, Any]], *, model: str = "unknown",
              config: str = "", dataset_version: str = "v1",
              k: int = DEFAULT_K) -> dict[str, Any]:
    """Roll per-case grades into the scorecard dict.

    Each per_case dict carries the grades that apply to it (absent keys are
    skipped, so a part# case only contributes to part_number_exact, etc.):
      {"part_number_exact": bool, "torque_exact": bool, "procedure_correct": bool,
       "retrieval_hit": bool, "hallucinated": bool, "refusal_correct": bool,
       "figure_correct": bool}
    'hallucinated' is True when the model asserted a value on a case where it
    should have refused (expect_refusal) — i.e. it answered when it shouldn't.
    """
    # counters: name -> [passed, n]
    counts: dict[str, list[int]] = {
        "part_number_exact": [0, 0],
        "torque_exact": [0, 0],
        "procedure_correct": [0, 0],
        "retrieval_hit_at_k": [0, 0],
        "refusal_correct": [0, 0],
        "figure_correct": [0, 0],
    }
    hall_pass, hall_n = 0, 0  # hallucination tracked separately (rate is a bare float)

    for c in per_case:
        for key, col in (
            ("part_number_exact", "part_number_exact"),
            ("torque_exact", "torque_exact"),
            ("procedure_correct", "procedure_correct"),
            ("retrieval_hit", "retrieval_hit_at_k"),
            ("refusal_correct", "refusal_correct"),
            ("figure_correct", "figure_correct"),
        ):
            if key in c and c[key] is not None:
                counts[col][1] += 1
                if c[key]:
                    counts[col][0] += 1
        if "hallucinated" in c and c["hallucinated"] is not None:
            hall_n += 1
            if not c["hallucinated"]:
                hall_pass += 1

    hallucination_rate = ((hall_n - hall_pass) / hall_n) if hall_n else 0.0

    metrics: dict[str, Any] = {
        "part_number_exact": _rate_block(*counts["part_number_exact"]),
        "torque_exact": _rate_block(*counts["torque_exact"]),
        "procedure_correct": _rate_block(*counts["procedure_correct"]),
        "retrieval_hit_at_k": _rate_block(*counts["retrieval_hit_at_k"], k=k),
        "hallucination_rate": hallucination_rate,
        # carry the observation count so passed() can fail a gate that was never
        # measured (n==0) instead of passing it vacuously at rate 0.0.
        "hallucination_n": hall_n,
        "refusal_correct": _rate_block(*counts["refusal_correct"]),
        "figure_correct": _rate_block(*counts["figure_correct"]),
    }

    scorecard = {
        "model": model,
        "config": config,
        "dataset_version": dataset_version,
        "n": len(per_case),
        "metrics": metrics,
        "gates": dict(GATES),
        "passed": False,
    }
    scorecard["passed"] = passed(scorecard)
    return scorecard


def _metric_rate(metrics: dict[str, Any], name: str) -> Optional[float]:
    """Pull the scalar rate for a metric out of the scorecard's metrics dict,
    whether it's a bare float (hallucination_rate) or a {'rate': ...} block."""
    v = metrics.get(name)
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("rate")
    return float(v)


def passed(scorecard: dict[str, Any]) -> bool:
    """True iff every GATE is cleared. A metric with no observations (n==0,
    rate 0.0) for a 'higher is better' gate fails — you cannot pass a gate you
    never measured."""
    m = scorecard.get("metrics", {})
    for name, gate in GATES.items():
        rate = _metric_rate(m, name)
        if rate is None:
            return False
        if _GATE_HIGHER_BETTER[name]:
            # an unmeasured higher-is-better gate (block n==0) must not pass
            block = m.get(name)
            if isinstance(block, dict) and block.get("n", 1) == 0:
                return False
            if rate < gate:
                return False
        else:
            # an unmeasured lower-is-better gate (no observations) must not pass either
            if name == "hallucination_rate" and m.get("hallucination_n", 0) == 0:
                return False
            if rate > gate:
                return False
    return True


# --------------------------------------------------------------------------- #
# Champion comparison (db.upsert_leaderboard(..., is_champion_if=is_champion_if))
# --------------------------------------------------------------------------- #
def _champion_score(metrics: dict[str, Any]) -> float:
    """Single weighted acceptance score for ranking candidates. Hallucination is
    weighted highest and enters as a PENALTY (1 - rate), so a model that invents
    values can never win on raw accuracy. Missing metrics contribute 0."""
    score = 0.0
    for name, w in _CHAMPION_WEIGHTS.items():
        rate = _metric_rate(metrics, name)
        if rate is None:
            continue
        if name == "hallucination_rate":
            score += w * (1.0 - rate)   # lower hallucination -> higher score
        else:
            score += w * rate
    return score


def is_champion_if(new: dict[str, Any], champ: Optional[dict[str, Any]]) -> bool:
    """Promotion rule for upsert_leaderboard. A new result becomes champion iff:
      1. it PASSES all acceptance gates (a failing model is never champion), AND
      2. there is no champion yet, OR it beats the champion on the weighted
         acceptance score (hallucination weighted highest).
    `new`/`champ` are the scorecard's 'metrics' dict OR a full scorecard; we
    accept either so callers can pass whichever they hold."""
    new_metrics = new.get("metrics", new) if isinstance(new, dict) else {}
    # Gate the candidate. If `new` is a full scorecard with 'passed', trust it;
    # otherwise compute it from the metrics block.
    if isinstance(new, dict) and "passed" in new:
        if not new["passed"]:
            return False
    else:
        if not passed({"metrics": new_metrics}):
            return False

    if not champ:
        return True
    champ_metrics = champ.get("metrics", champ) if isinstance(champ, dict) else {}
    return _champion_score(new_metrics) > _champion_score(champ_metrics)


if __name__ == "__main__":
    # bare-Mac self-check: pure graders + aggregate + champion logic
    assert part_number_exact("part number is 30650014 (VCC-1)", "30 650 014")
    assert not part_number_exact("no number here", "30650014")
    assert torque_exact("Torque is 15 Nm (VCC-112365-1)", 15)
    assert torque_exact("tighten to 15N·m", 15.0)
    assert not torque_exact("tighten to 20 Nm", 15)
    assert detect_refusal("I couldn't find a verified value for this.")
    assert detect_refusal("Which vehicle is this for? Give me the VIN.")
    assert not detect_refusal("15 Nm (VCC-112365-1), for the B5254T2.")

    # part-number gate: not fooled by citations or substrings; still accepts a 5-digit part
    assert not part_number_exact("The part is 30650999 (VCC-30650014-1).", "30650014")
    assert not part_number_exact("the part is 12345678", "1234567")
    assert part_number_exact("part number 30650 fits", "30650")
    # refusal vs grounded: a grounded answer mentioning model/year is NOT a refusal
    assert not detect_refusal("Generally it is 15 Nm, though your model and year affect it.")
    assert detect_refusal("Nothing in the verified sources covers that.")
    assert detect_refusal("I need the car first — what's the VIN, engine code, or model and year?")
    assert asserts_value("The torque is 15 Nm") and asserts_value("the part is 30650014")
    assert not asserts_value("I couldn't find a verified value (VCC-112365-1).")

    per_case = [
        {"part_number_exact": True, "retrieval_hit": True},
        {"torque_exact": True, "retrieval_hit": True},
        {"refusal_correct": True, "hallucinated": False},
        {"procedure_correct": True, "retrieval_hit": True},
    ]
    sc = aggregate(per_case, model="qwen3-4b", config="r16")
    assert sc["n"] == 4
    assert sc["metrics"]["part_number_exact"]["rate"] == 1.0
    assert sc["metrics"]["hallucination_rate"] == 0.0
    # not passed: torque/part have n=1 each at 1.0 but refusal/retrieval gates met;
    # however gates with n==0 for some metrics -> ensure 'passed' is a bool
    assert isinstance(sc["passed"], bool)

    # champion: a perfect-on-all gated scorecard beats none, and a failing one loses
    good = aggregate([
        {"part_number_exact": True, "torque_exact": True, "refusal_correct": True,
         "retrieval_hit": True, "hallucinated": False, "procedure_correct": True},
    ])
    # n is tiny so gates with the same case satisfy >= thresholds (rate 1.0)
    assert is_champion_if(good, None) is good["passed"]
    bad = dict(good)
    bad["passed"] = False
    assert is_champion_if(bad, good) is False

    # a scorecard with ZERO hallucination observations must NOT pass the safety gate
    sc_no_hall = aggregate([
        {"part_number_exact": True, "torque_exact": True, "refusal_correct": True,
         "retrieval_hit": True, "procedure_correct": True},  # note: no 'hallucinated' key
    ])
    assert sc_no_hall["metrics"]["hallucination_n"] == 0
    assert sc_no_hall["passed"] is False, "vacuous hallucination gate must fail at n==0"
    print("OK metrics self-check")
