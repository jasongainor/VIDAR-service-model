"""grounding/contract.py — the ONE shared contract for the VIDA grounded model.

The model is trained on, and served with, the SAME message shape produced here.
Imported by:
  - dataset/      (training-example generation)
  - vida-eval/    (benchmark generation + scoring)
  - the serving layer (Open WebUI / MCP system prompt)

Design decisions (locked 2026-06-20 — see docs/DECISIONS.md):
  * FUNCTION-CALLING agent shape, NOT a pre-injected [CONTEXT] block. The model
    emits tool_calls, reads role:"tool" JSON results, then answers with citations.
  * STICKY VEHICLE: once a vehicle is established in the conversation it is
    inherited by every later turn. The model ELICITS the vehicle when missing and
    still works without one (vehicle-agnostic questions answered directly).
  * NO TOPIC BLEED: the vehicle (and the active component on "while I'm in there")
    persists; the topic does not. Topic A's answer must never contaminate topic B.
  * CITATION TOKEN = the real identifier the server can emit and eval can resolve:
    vida_doc_ref (e.g. VCC-118802-1), falling back to doc_id. No invented ids.
  * CROSS-VEHICLE = FALLBACK only: with a car in context, answer for that car;
    surface other-vehicle results only on a this-car miss / weak car description.
  * IMAGES = figure-retrieval only ("show me the figure for X"); NO callout->part
    (that mapping does not exist in the data — see docs/DECISIONS.md).

Pure stdlib (json, re, dataclasses) so it runs anywhere with no extra deps.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

CONTRACT_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Example types + target mix (shares are the generation target; tune from eval)
# --------------------------------------------------------------------------- #
EXAMPLE_TYPES: dict[str, dict[str, Any]] = {
    "grounded":     {"share": 0.46, "purpose": "call a tool, read the result, cite the source"},
    "refusal":      {"share": 0.20, "purpose": "tool returns nothing usable -> refuse, don't invent"},
    "elicitation":  {"share": 0.10, "purpose": "no vehicle given -> ask for VIN/engine/model+year"},
    "cross_vehicle":{"share": 0.10, "purpose": "this-car miss -> surface other-vehicle, flag verify"},
    "multi_turn":   {"share": 0.10, "purpose": "sticky vehicle across turns; no topic bleed"},
    "figure":       {"share": 0.04, "purpose": "show the figure(s) for X (figure-retrieval only)"},
}
# Tool-routing (epc_part vs search_docs) is exercised within grounded/part examples.

ANSWER_TYPES = ("torque", "part_number", "procedure", "pinout", "fluid", "figure",
                "refusal", "elicitation", "cross_vehicle", "other")

# --------------------------------------------------------------------------- #
# System prompt — the behavioral contract the model is trained to follow
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are a Volvo VIDA repair assistant. You answer ONLY from the VIDA knowledge-base \
tools; you never answer car facts from memory. A wrong torque, wire color, fluid \
capacity, or part number can damage a car or injure someone — "I could not verify \
that" is a correct answer, inventing a value is a failure.

TOOLS (call them; do not guess):
- search_docs(query, car?) — hybrid search over procedures/specs/wiring. Returns \
ranked snippets with a source id and a scope ('this car' | 'other vehicles'). Open \
the full text with get_document before quoting a value.
- get_document(doc_id) — full text of one document.
- epc_part(part_number? , component?) — EXACT parts catalogue lookup. Use this (NOT \
search_docs) when given a specific part number, or asked for the part number(s) of a \
named component (oxygen sensor, thermostat, ...). Returns description + applicability \
scope.
- lookup_pin(module?, pin?, query?) — wiring / connector pinouts.
- get_figures(query?, doc_id?) — relevant figure(s) for a procedure. You can show a \
figure; you CANNOT name a callout->part number (that data does not exist) — describe \
the figure and cite its document instead.

VEHICLE CONTEXT (sticky):
- The vehicle is whatever the user established earlier in THIS conversation (a VIN, \
engine code, or model+year), or a [VEHICLE]...[/VEHICLE] block in their message. \
Once set, it carries to every later turn — "what about the exhaust manifold?" reuses \
the same car. Do not ask again once you have it.
- If a vehicle-specific question arrives and NO vehicle is established, ask for the \
VIN, engine code, or model and year before answering — do not guess a vehicle. \
Answer vehicle-agnostic questions directly without asking.
- Pass car=<engine code> to search_docs only to deliberately retarget another vehicle.

NO TOPIC BLEED: a new question is a NEW topic unless it explicitly refers back \
("it", "that one", "while I'm in there"). Build the tool query from the latest \
question's words only; never mix in a component or system from an earlier turn. The \
VEHICLE persists across turns; the TOPIC does not.

ANSWERING:
- Lead with the answer, then cite the source id(s) in parentheses, then state scope \
if it is not this car. Example: "15 Nm (VCC-112365-1), for the B5254T2."
- For part numbers, route to epc_part and prefer 'this car' parts; say so.
- CROSS-VEHICLE IS A FALLBACK: with a vehicle in context, answer for that car and do \
not volunteer other vehicles. Only if this car yields no answer (or its description \
is too weak to be sure) may you surface an other-vehicle result — and then flag it: \
"appears for a related vehicle — verify before use".
- If the tools return nothing that answers the question, refuse: say you could not \
find a verified value and ask for the VIN / engine code / model+year to narrow it.
- If sources conflict on a value, present the conflict and prefer this car's source; \
do not silently pick one.
"""

# --------------------------------------------------------------------------- #
# Serving-time P2-default profile (see docs/DECISIONS.md D19).
# The TRAINING contract above elicits the vehicle when none is given. In real use the
# operator is almost always on a Volvo P2-platform car (this build's default is the
# 2005 S60R / B5254T4), and being asked for a VIN — which neither the model nor the
# tools can decode (there is no VIN decoder in src/server.py) — is a dead end that just
# stalls the first turn. This SERVED prompt keeps every safety rule intact
# (retrieve-or-refuse, cite the source, cross-vehicle flagging, no topic bleed) but
# DEFAULTS the platform and takes a first grounded pass instead of interrogating the
# user. To be folded back into the elicitation training examples on the next run so
# train==serve is restored.
# --------------------------------------------------------------------------- #
_P2_VEHICLE_BLOCK = """VEHICLE CONTEXT (Volvo P2 platform — assume it):
- You work on Volvo P2-platform cars (2001–2009: S60, S80, V70, XC70, XC90, and the S60R/V70R). This build's default car is the 2005 Volvo S60R (engine B5254T4), and most P2 procedures and specs are shared across the platform. Unless the user names another vehicle, ASSUME the P2 platform and answer for it.
- DO NOT ask for a VIN or FIN — neither you nor the tools can decode one, so it is a dead end. Never open a reply with "which vehicle is this?".
- On the FIRST vehicle-specific question with no vehicle stated, take a stab: call search_docs (it already scopes to the P2 default and blends shared-platform sources), open the top hit with get_document, and answer with the citation, stating which vehicle/engine the source applies to (e.g. "15 Nm (VCC-112365-1), for the B5254T2"). The user sees that applicability and can correct you.
- Only AFTER that first grounded pass, and only if it actually matters (the value genuinely differs across P2 variants, or the user needs a variant your source doesn't cover), ask them to narrow it — by ENGINE CODE or MODEL+YEAR (never a VIN).
- STICKY: once the user gives an engine code or model+year, reuse it for every later turn; don't ask again.
- Pass car=<engine code> to search_docs only to deliberately retarget a different vehicle."""


def _serving_prompt() -> str:
    """SYSTEM_PROMPT with the elicit-first VEHICLE CONTEXT swapped for the P2-default
    take-a-stab block, and the retrieval-miss fallback pointed at engine-code/model+year
    instead of a VIN. Sliced on unique anchors so inner wording can change freely."""
    p = SYSTEM_PROMPT
    start = p.index("VEHICLE CONTEXT (sticky):")
    end = p.index("NO TOPIC BLEED:")
    p = p[:start] + _P2_VEHICLE_BLOCK + "\n\n" + p[end:]
    p = p.replace(
        "ask for the VIN / engine code / model+year to narrow it.",
        "ask for the engine code or model+year (never a VIN) to narrow it. Never invent a value.",
    )
    return p


SERVING_SYSTEM_PROMPT = _serving_prompt()

# --------------------------------------------------------------------------- #
# Tool schemas (OpenAI function-calling shape). These are the model-facing FACT
# tools and they mirror the live MCP server's fact tools (src/server.py), with two
# deliberate differences: (1) list_sources is intentionally EXCLUDED — it is an
# operator/inventory tool, not something the model should call to answer a repair
# question; (2) search_docs exposes a friendly `car` (engine-code retarget) and hides
# the server's `profile`/`limit` knobs, which the serving layer manages. So
# TOOL_NAMES is a curated SUBSET of the server's @_tool set, not an equality.
# --------------------------------------------------------------------------- #
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": ("Hybrid FTS+vector search over VIDA procedures, "
                            "specifications and wiring. Returns ranked snippets, each "
                            "with a source id and scope. Use for procedures and "
                            "descriptive/spec text, including 'Tightening torque' tables."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search terms from the latest question only"},
                    "car": {"type": "string", "description": "optional engine code to retarget another vehicle"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Full normalized text of one document, by doc_id or VCC number.",
            "parameters": {
                "type": "object",
                "properties": {"doc_id": {"type": "string"}},
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "epc_part",
            "description": ("Exact parts-catalogue lookup. Use for a specific part "
                            "number, or to find the part number(s) of a named component. "
                            "Returns description + applicability scope per part."),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_number": {"type": "string"},
                    "component": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_pin",
            "description": "Wiring / connector pinout lookup by module, pin, or free-text query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {"type": "string"},
                    "pin": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_figures",
            "description": ("Relevant figure(s) for a procedure/component. Returns image "
                            "ids + the document they belong to. Figure-retrieval only — "
                            "there is no callout->part-number mapping."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "doc_id": {"type": "string"},
                },
                "required": [],
            },
        },
    },
]

TOOL_NAMES = tuple(t["function"]["name"] for t in TOOLS)

# --------------------------------------------------------------------------- #
# Vehicle
# --------------------------------------------------------------------------- #
@dataclass
class Vehicle:
    year: Optional[str] = None
    model: Optional[str] = None
    engine: Optional[str] = None
    vin: Optional[str] = None

    def label(self) -> str:
        parts = []
        ym = " ".join(x for x in (str(self.year) if self.year else None, self.model) if x)
        if ym:
            parts.append(ym)
        if self.engine:
            parts.append(self.engine)
        if self.vin:
            parts.append(f"VIN {self.vin}")
        return " | ".join(parts) if parts else "unspecified vehicle"

    def vehicle_block(self) -> str:
        return f"[VEHICLE] {self.label()} [/VEHICLE]"

    def is_empty(self) -> bool:
        return not any((self.year, self.model, self.engine, self.vin))

    def to_meta(self) -> dict[str, Any]:
        return {"year": self.year, "model": self.model, "engine": self.engine, "vin": self.vin}


# --------------------------------------------------------------------------- #
# Citations + refusal/cross-vehicle/elicitation/figure templates
# --------------------------------------------------------------------------- #
def citation(vida_doc_ref: Optional[str] = None, doc_id: Optional[str] = None) -> str:
    """The single citation token. Prefer the human VCC ref; fall back to doc_id.
    eval resolves these back against documents.vida_doc_ref / documents.doc_id."""
    tok = vida_doc_ref or doc_id or "unknown"
    return f"({tok})"


REFUSAL_TEMPLATE = (
    "I couldn't find a verified value for this in the available data. "
    "Give me the VIN, engine code, or the model and year so I can narrow the search."
)

# Variants so ~20% of the dataset is not one literal sentence (the model must learn
# "retrieval came up empty -> refuse", not memorize a string). Each MUST contain a
# refusal phrase metrics.detect_refusal recognizes. See docs/AUDIT-2026-06-20.md.
REFUSAL_VARIANTS = (
    REFUSAL_TEMPLATE,
    "I don't have a verified figure for that in the data I can see, and I won't guess. "
    "Give me the VIN, engine code, or model and year and I'll dig further.",
    "Nothing in the verified sources covers that. Tell me the VIN, engine code, or the "
    "model and year so I can narrow it down.",
    "I can't confirm that value from the VIDA data here. Share the VIN, engine code, or "
    "model and year and I'll re-check rather than give you an unverified number.",
)

ELICITATION_TEMPLATE = (
    "Which vehicle is this for? Give me the VIN, the engine code, or the model and "
    "year and I'll pull the verified figures for it."
)

ELICITATION_VARIANTS = (
    ELICITATION_TEMPLATE,
    "Which vehicle are we working on? Give me the VIN, the engine code, or the model "
    "and year and I'll pull the verified figures.",
    "I need the car first — what's the VIN, engine code, or model and year? Then I'll "
    "look up the verified values for it.",
    "Tell me which vehicle this is (the VIN, engine code, or model and year) and I'll "
    "get the exact figures.",
)


def _pick(variants: tuple[str, ...], seed: Optional[str]) -> str:
    """Deterministic variant choice from a stable string seed (e.g. a split_key), so
    generation stays reproducible and a record always renders the same way."""
    if not seed:
        return variants[0]
    h = 0
    for ch in str(seed):
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return variants[h % len(variants)]


def refusal_text(seed: Optional[str] = None) -> str:
    return _pick(REFUSAL_VARIANTS, seed)


def elicitation_text(seed: Optional[str] = None) -> str:
    return _pick(ELICITATION_VARIANTS, seed)


def cross_vehicle_answer(value_sentence: str, ref: str, related_desc: str,
                         confidence: str = "medium") -> str:
    """e.g. cross_vehicle_answer('Torque is 20 Nm', 'VCC-143022-1',
    'the 2003 V70 (shared engine B5254T2)')

    `ref` is stated in plain prose, NOT wrapped in citation() — citation() implies a VIDA
    document reference (a "VCC-..." id), but build_cross_vehicle passes a raw Volvo part
    number here. Wrapping it made the model generalize a "(VCC-...)" look onto the part
    number itself (confusing for a real user: it looks like a doc ref, not an order
    number) and defeated the eval grader's anti-cheat citation-stripping check, which
    correctly strips "(VCC-...)"-shaped spans before matching a part number — so the one
    place the right answer lived got stripped. Confirmed 2026-07-04 in a retrain eval:
    2 of 10 cross_vehicle cases had the exact right part number, scored wrong by this."""
    return (f"No exact match for this car. {value_sentence}, part number {ref}, "
            f"appears for {related_desc} — likely compatible, verify before use. "
            f"[confidence: {confidence}]")


def figure_answer(doc_ref: str, n_figures: int, what: str) -> str:
    fig = "figure" if n_figures == 1 else f"{n_figures} figures"
    return (f"Showing the {fig} for {what} from {citation(doc_ref)}. "
            f"I can describe what's pictured, but VIDA does not expose a "
            f"callout-number→part-number mapping, so I can't label the balloons.")


# --------------------------------------------------------------------------- #
# Tool-RESULT envelopes — mirror src/server.py's ACTUAL return shapes so the
# trained tool-result structure == the served one (and == what vida-eval feeds the
# model at grade time). Audit 2026-06-20: training previously wrapped every result in
# {"results": [...]}, a shape the live MCP server never produces, breaking train==serve.
#   search_docs / lookup_pin -> a BARE LIST of row dicts
#   epc_part                  -> {"query": {...}, "rows": [...]}
#   get_figures               -> a summary dict {doc_id, title, scope, figures, ...}
# These helpers are the single source of that shape; dataset/ and vida-eval/ both call
# them so the three paths can never drift again.
# --------------------------------------------------------------------------- #
def search_docs_result(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(rows)


def lookup_pin_result(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(rows)


def epc_part_result(rows: list[dict[str, Any]], *, part_number: Optional[str] = None,
                    component: Optional[str] = None) -> dict[str, Any]:
    return {"query": {"part_number": part_number, "component": component}, "rows": list(rows)}


def get_figures_result(*, doc_id: str, title: Optional[str], figures: list[Any],
                       scope: str = "this car", vida_doc_ref: Optional[str] = None,
                       note: Optional[str] = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "doc_id": doc_id, "title": title, "scope": scope,
        "figures_shown_to_user": len(figures), "figures": list(figures),
    }
    # The live server's get_figures summary does NOT carry the VCC ref; we add it here
    # (a faithful superset) so the model has a citation token for the figure's document.
    if vida_doc_ref:
        out["vida_doc_ref"] = vida_doc_ref
    if note:
        out["note"] = note
    return out


# --------------------------------------------------------------------------- #
# Message + record builders (OpenAI chat shape; Unsloth/transformers compatible)
# --------------------------------------------------------------------------- #
def system_msg() -> dict[str, Any]:
    return {"role": "system", "content": SYSTEM_PROMPT}


def serving_system_msg() -> dict[str, Any]:
    """The P2-default 'take a stab' variant used by the served model (see D19)."""
    return {"role": "system", "content": SERVING_SYSTEM_PROMPT}


def user_msg(text: str, vehicle: Optional[Vehicle] = None) -> dict[str, Any]:
    """If a vehicle is supplied AND non-empty, prepend its [VEHICLE] block (this is
    how a user 'establishes' the car in turn 1). Later turns omit it (sticky)."""
    if vehicle is not None and not vehicle.is_empty():
        return {"role": "user", "content": f"{vehicle.vehicle_block()}\n{text}"}
    return {"role": "user", "content": text}


def assistant_tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    assert name in TOOL_NAMES, f"unknown tool {name!r}"
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }],
    }


def tool_result(call_id: str, content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def assistant_answer(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def call_id(seq: int) -> str:
    return f"call_{seq:04d}"


# --------------------------------------------------------------------------- #
# Canonical training record
# --------------------------------------------------------------------------- #
@dataclass
class Record:
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"messages": self.messages, "metadata": self.metadata},
                          ensure_ascii=False)


def make_record(messages: list[dict[str, Any]], *, example_type: str,
                answer_type: Optional[str] = None, vehicle: Optional[Vehicle] = None,
                split_key: Optional[str] = None,
                source_ids: Optional[list[str]] = None) -> Record:
    assert example_type in EXAMPLE_TYPES, f"unknown example_type {example_type!r}"
    if answer_type is not None:
        assert answer_type in ANSWER_TYPES, f"unknown answer_type {answer_type!r}"
    meta = {
        "contract_version": CONTRACT_VERSION,
        "example_type": example_type,
        "answer_type": answer_type,
        "vehicle": vehicle.to_meta() if vehicle else None,
        "split_key": split_key,
        "source_ids": source_ids or [],
    }
    return Record(messages=messages, metadata=meta)


# --------------------------------------------------------------------------- #
# Light validation used by tests / generators / the format smoke-test
# --------------------------------------------------------------------------- #
def validate_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Return a list of problems (empty == valid). Enforces the shape the chat
    template must round-trip: system first, user/assistant/tool roles only,
    tool_calls answered by matching tool_call_id, ends on an assistant answer."""
    problems: list[str] = []
    if not messages:
        return ["empty messages"]
    if messages[0].get("role") != "system":
        problems.append("first message is not system")
    open_calls: set[str] = set()
    for i, m in enumerate(messages):
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            problems.append(f"msg {i}: bad role {role!r}")
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                open_calls.add(tc["id"])
                fn = tc.get("function", {})
                if fn.get("name") not in TOOL_NAMES:
                    problems.append(f"msg {i}: unknown tool {fn.get('name')!r}")
                try:
                    json.loads(fn.get("arguments", "{}"))
                except Exception:
                    problems.append(f"msg {i}: tool arguments not valid JSON")
        if role == "tool":
            tcid = m.get("tool_call_id")
            if tcid not in open_calls:
                problems.append(f"msg {i}: tool_result for unknown call {tcid!r}")
            else:
                open_calls.discard(tcid)
    if open_calls:
        problems.append(f"unanswered tool_calls: {sorted(open_calls)}")
    if messages[-1].get("role") != "assistant" or messages[-1].get("content") in (None, ""):
        problems.append("conversation does not end on a non-empty assistant answer")
    return problems


if __name__ == "__main__":
    # tiny self-check
    v = Vehicle(year="2004", model="XC70", engine="B5254T2")
    msgs = [
        system_msg(),
        user_msg("Torque for the intake manifold bolts?", vehicle=v),
        assistant_tool_call("search_docs", {"query": "intake manifold tightening torque"}, call_id(1)),
        tool_result(call_id(1), search_docs_result([{"doc_id": "DiagSwdl:Document:-112365",
                                                     "vida_doc_ref": "VCC-112365-1",
                                                     "snippet": "Tightening torque: 15 Nm",
                                                     "scope": "this car"}])),
        assistant_answer(f"15 Nm {citation('VCC-112365-1')}, for the B5254T2."),
    ]
    rec = make_record(msgs, example_type="grounded", answer_type="torque", vehicle=v,
                      split_key="XC70|torque", source_ids=["VCC-112365-1"])
    probs = validate_messages(rec.messages)
    print("OK" if not probs else probs)
    print(rec.to_json()[:200], "...")

    # envelope conformance: shapes must match the server (list / {"rows"} / figures dict)
    assert isinstance(search_docs_result([{"doc_id": "d"}]), list)
    assert isinstance(lookup_pin_result([{"pin": "1"}]), list)
    assert "rows" in epc_part_result([{"part_number": "1"}], part_number="1")
    assert get_figures_result(doc_id="d", title="t", figures=["a"])["doc_id"] == "d"
    # variants stay reproducible and distinct
    assert refusal_text("XC70|refusal") in REFUSAL_VARIANTS
    assert refusal_text("a") == refusal_text("a")
    # validate_messages must not KeyError on a message missing 'role'
    assert validate_messages([{"content": "x"}]) and \
        "first message is not system" in validate_messages([{"content": "x"}])
    print("contract.py self-check OK")
