"""dataset/examples.py — turn real VIDA rows into grounding.contract Records.

One builder per example type in gc.EXAMPLE_TYPES (DECISIONS D1–D7). Every builder
returns a gc.Record whose messages are the EXACT function-calling shape the model is
served under (system -> user -> assistant tool_calls -> role:"tool" JSON -> grounded
answer), and every value/id/vehicle is pulled from the DB (no invented facts, D7).
Builders return None when the row can't make a clean example (e.g. no torque parses),
so the generator just skips it.

All synthesized tool_result JSON mirrors what src/server.py actually returns, so the
trained tool-result shape == the served one. Pure stdlib.
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Callable, Optional

from grounding import contract as gc

from . import sources as src
from .torque import first_torque


# --------------------------------------------------------------------------- #
# Live-retrieval injection (D-realism). The grounded builders below want the
# tool-result turn to be the EXACT output the model meets at serve time — the
# real, multi-hit, noisy `search_docs` list — not a hand-distilled single hit.
# But this module must stay import-clean + stdlib-only on a bare Mac (the
# py_compile gate and the no-DB self-check), and the real retriever pulls in the
# MCP server + an embeddings backend. So we INJECT it: gen_dataset wires the live
# `server.search_docs` via set_retriever(); when it's absent (self-check) the
# builders fall back to the synthetic single-hit result they used before.
# --------------------------------------------------------------------------- #
_SEARCH: Optional[Callable[..., list[dict]]] = None
_GETDOC: Optional[Callable[..., dict]] = None


def set_retriever(search: Optional[Callable[..., list[dict]]] = None,
                  get_document: Optional[Callable[..., dict]] = None) -> None:
    """Wire (or clear) the live retriever. `search(query, car=...)` must return the
    SAME object src/server.py's search_docs returns (a bare list of hit dicts);
    `get_document(doc_id)` the same dict src/server.py's get_document returns. Both are
    used so the recorded tool turns are byte-identical to the serve path."""
    global _SEARCH, _GETDOC
    _SEARCH = search
    _GETDOC = get_document


def using_live_retrieval() -> bool:
    return _SEARCH is not None


def _live_search(query: str, car: Optional[str]) -> Optional[list[dict]]:
    """Real search_docs output, or None when no retriever is wired / it errors."""
    if _SEARCH is None:
        return None
    try:
        res = _SEARCH(query, car=car)
    except Exception:
        return None
    return res if isinstance(res, list) else None


def _live_getdoc(doc_id: str) -> Optional[dict]:
    """Real get_document output, or None when no retriever is wired / it errors."""
    if _GETDOC is None:
        return None
    try:
        res = _GETDOC(doc_id)
    except Exception:
        return None
    return res if isinstance(res, dict) else None


def _find_hit(results: Optional[list[dict]], doc: src.Doc) -> tuple[Optional[int], Optional[dict]]:
    """Locate `doc` inside live results (by doc_id, then vida_doc_ref). Returns
    (rank, hit) or (None, None) — a genuine retrieval miss the caller must skip,
    so we never assert grounding the retriever didn't actually surface."""
    for i, r in enumerate(results or []):
        if not isinstance(r, dict):
            continue
        if r.get("doc_id") == doc.doc_id:
            return i, r
        if doc.vida_doc_ref and r.get("vida_doc_ref") == doc.vida_doc_ref:
            return i, r
    return None, None


def _value_visible(payload: Any, value_nm: float) -> bool:
    """True iff the torque value is literally readable in the tool payload (so the
    answer is grounded in what the model can SEE, not memorized). The lookbehind
    rejects a digit/letter/dot immediately before the value, so a part-number tail
    ('PN400 Nm') or a decimal ('1.400') never masquerades as a 400 Nm torque spec."""
    blob = json.dumps(payload, ensure_ascii=False)
    v = re.escape(f"{value_nm:g}")
    return bool(re.search(rf"(?<![\w.]){v}\s*(?:Nm|N·m|N\.?m|N m)", blob, re.I))


# Deterministic phrasing variety (keyed on doc_id via gc._pick) so the model learns
# the BEHAVIOR robustly across phrasings rather than one literal template — same
# example count, more diversity. No invented facts; only the wording rotates.
_TORQUE_Q = (
    "What's the tightening torque for {c}?",
    "What torque should I use for {c}?",
    "Tightening torque for {c}?",
)
_TORQUE_QUERY = (
    "{c} tightening torque",
    "tightening torque for {c}",
)


def _topic_from_title(title: Optional[str], fallback: str = "this component") -> str:
    """A natural-language topic from a doc title, e.g. 'Replacing the thermostat'
    -> 'the thermostat'."""
    if not title:
        return fallback
    t = title.strip()
    for verb in ("Replacing", "Removing", "Installing", "Checking", "Replacement of"):
        if t.startswith(verb):
            return t[len(verb):].strip().lstrip("- ").strip() or fallback
    return t


# --------------------------------------------------------------------------- #
# 1. grounded — torque / part / procedure / pinout (call a tool, cite the source)
# --------------------------------------------------------------------------- #
def build_grounded_torque(doc: src.Doc) -> Optional[gc.Record]:
    tq = first_torque(doc.text_md or "")
    if tq is None:
        return None
    veh = src.pick_vehicle_for_doc(doc)
    comp = tq.component
    q = gc._pick(_TORQUE_Q, doc.doc_id).format(c=comp)
    query = gc._pick(_TORQUE_QUERY, doc.doc_id).format(c=comp)
    cid = gc.call_id(1)
    val = f"{tq.value_nm:g} {tq.unit}"
    tail = f", for the {veh.engine}." if veh.engine else "."

    # Retrieve EXACTLY the way the model calls at serve: query only, no car retarget
    # (the served model emits {"query": ...}). This keeps the recorded tool_call
    # self-consistent with the result it shows AND makes train == serve byte-for-byte.
    # Retrieve EXACTLY the way the model calls at serve: query only, no car retarget.
    results = _live_search(query, None)
    if results is not None:
        # Realism path: train on the EXACT noisy multi-hit list the model meets at serve
        # time. Ground ONLY if the gold doc is actually retrieved (often NOT rank 0 —
        # the hard-read signal the old single-hit data lacked); else skip.
        _, hit = _find_hit(results, doc)
        if hit is None:
            return None
        cite_ref = hit.get("vida_doc_ref") or doc.vida_doc_ref
        cite_id = hit.get("doc_id") or doc.doc_id
        msgs = [
            gc.system_msg(),
            gc.user_msg(q, vehicle=veh),
            gc.assistant_tool_call("search_docs", {"query": query}, cid),
            gc.tool_result(cid, results),
        ]
        # The value must be visible in the SPECIFIC cited hit (not a neighbour — that
        # would be cross-hit hallucination). If the search SNIPPET doesn't show it, open
        # the cited doc (the real search->read->answer flow; _dispatch_tool serves
        # get_document identically). Skip if it still can't be exposed or the doc is too
        # large for the training window.
        if not _value_visible(hit, tq.value_nm):
            full = _live_getdoc(hit.get("doc_id") or doc.doc_id)
            if not full or not _value_visible(full, tq.value_nm) or len(json.dumps(full)) > 9000:
                return None
            cid2 = gc.call_id(2)
            msgs.append(gc.assistant_tool_call("get_document", {"doc_id": cite_id}, cid2))
            msgs.append(gc.tool_result(cid2, full))
        msgs.append(gc.assistant_answer(f"{val} {gc.citation(cite_ref, cite_id)}{tail}"))
    else:
        # Bare-Mac self-check fallback: synthetic single clean hit (no server needed).
        msgs = [
            gc.system_msg(),
            gc.user_msg(q, vehicle=veh),
            gc.assistant_tool_call("search_docs", {"query": query}, cid),
            gc.tool_result(cid, gc.search_docs_result([{
                "doc_id": doc.doc_id, "vida_doc_ref": doc.ref(),
                "title": doc.title, "snippet": f"{comp}: {val}", "scope": "this car",
            }])),
            gc.assistant_answer(f"{val} {gc.citation(doc.vida_doc_ref, doc.doc_id)}{tail}"),
        ]
    return gc.make_record(msgs, example_type="grounded", answer_type="torque",
                          vehicle=veh, split_key=f"{veh.model or '?'}|torque",
                          source_ids=[doc.ref()])


def build_grounded_part(conn: sqlite3.Connection, part: src.Part) -> Optional[gc.Record]:
    profs = src.profiles_for_part(conn, part.part_number, limit=5)
    veh = src.vehicle_from_profile(*profs[0]) if profs else gc.Vehicle()
    scope = "this car" if profs else "not vehicle-specific"
    q = f"Look up Volvo part number {part.part_number} — what is it?"
    cid = gc.call_id(1)
    msgs = [
        gc.system_msg(),
        gc.user_msg(q, vehicle=veh if not veh.is_empty() else None),
        gc.assistant_tool_call("epc_part", {"part_number": part.part_number}, cid),
        gc.tool_result(cid, gc.epc_part_result([{
            "part_number": part.part_number, "description": part.description,
            "section": part.section_title, "scope": scope,
            "citation": f"EPC/PartItems/{part.part_number}",
        }], part_number=part.part_number)),
        gc.assistant_answer(
            f"{part.part_number} is the {part.description} ({scope})."),
    ]
    return gc.make_record(msgs, example_type="grounded", answer_type="part_number",
                          vehicle=veh if not veh.is_empty() else None,
                          split_key=f"{veh.model or 'any'}|part",
                          source_ids=[part.part_number])


def build_grounded_procedure(doc: src.Doc) -> Optional[gc.Record]:
    if not doc.text_md or len(doc.text_md) < 120:
        return None
    veh = src.pick_vehicle_for_doc(doc)
    topic = _topic_from_title(doc.title)
    q = f"How do I {doc.title[0].lower() + doc.title[1:]}?" if doc.title else f"How do I service {topic}?"
    query = doc.title or topic
    cid = gc.call_id(1)

    results = _live_search(query, None)  # query-only, matching serve-time calls
    if results is not None:
        # Realism path: real multi-hit list. Answer is grounded in the retrieved
        # gold-doc snippet (visible in the tool result), citing the retrieved ref.
        _, hit = _find_hit(results, doc)
        if hit is None:
            return None
        snippet = " ".join(str(hit.get("snippet") or hit.get("lead") or "").split())[:400]
        if len(snippet) < 30:
            return None  # nothing quotable surfaced -> skip
        cite_ref = hit.get("vida_doc_ref") or doc.vida_doc_ref
        cite_id = hit.get("doc_id") or doc.doc_id
        tool_payload: Any = results
    else:
        snippet = " ".join((doc.text_md or "").split())[:400]
        cite_ref, cite_id = doc.vida_doc_ref, doc.doc_id
        tool_payload = gc.search_docs_result([{
            "doc_id": doc.doc_id, "vida_doc_ref": doc.ref(),
            "title": doc.title, "snippet": snippet, "scope": "this car",
        }])

    msgs = [
        gc.system_msg(),
        gc.user_msg(q, vehicle=veh),
        gc.assistant_tool_call("search_docs", {"query": query}, cid),
        gc.tool_result(cid, tool_payload),
        gc.assistant_answer(
            f"See {doc.title!r} {gc.citation(cite_ref, cite_id)}: {snippet[:200]}"
            + ("…" if len(snippet) > 200 else "")),
    ]
    return gc.make_record(msgs, example_type="grounded", answer_type="procedure",
                          vehicle=veh, split_key=f"{veh.model or '?'}|procedure",
                          source_ids=[doc.ref()])


def build_grounded_pinout(row: sqlite3.Row) -> Optional[gc.Record]:
    module = row["module"] or "the module"
    pin = row["pin"] or "?"
    function = row["function"]
    wire = row["wire_color"]
    if not function:
        return None
    veh = src.S60R_PINOUT_VEHICLE
    src_ref = f"{row['source_file']}:{row['csv_line']}"
    connector = row["connector"]
    goes_to = row["goes_to"]
    q = f"What is connected to {module} pin {pin}, and what wire color is it?"
    cid = gc.call_id(1)
    ans = f"{module} pin {pin}: {function}"
    if wire:
        ans += f", wire color {wire}"
    ans += f" {gc.citation(src_ref)}."
    msgs = [
        gc.system_msg(),
        gc.user_msg(q, vehicle=veh),
        gc.assistant_tool_call("lookup_pin", {"module": module, "pin": str(pin)}, cid),
        gc.tool_result(cid, gc.lookup_pin_result([{
            "module": module, "connector": connector, "pin": pin, "wire_color": wire,
            "goes_to": goes_to, "function": function,
            "citation": f"{module} {connector or ''}:{pin} — {src_ref}".strip(),
            "source_row": src_ref,
        }])),
        gc.assistant_answer(ans),
    ]
    return gc.make_record(msgs, example_type="grounded", answer_type="pinout",
                          vehicle=veh, split_key="S60R|pinout", source_ids=[src_ref])


# --------------------------------------------------------------------------- #
# 2. refusal — tool returns nothing usable -> refuse, don't invent (D1)
# --------------------------------------------------------------------------- #
# A spread of plausibly-unanswerable specific-value asks so refusal generalizes to
# "retrieval returned nothing usable -> refuse", not one memorized sentence. Each is a
# (question, search-query, answerable_marker) triple across value TYPES
# (torque/fluid/wire/part). `answerable_marker` is a regex that, if it matches the REAL
# retrieved results, means the corpus probably DOES answer the ask (the distinctive
# component co-occurs with a value of the asked type) — so we SKIP rather than train a
# contradictory "results contain the value but refuse" example. Over-skipping is safe
# (just fewer refusals); a contradiction is not.
#
# `answerable_marker` must be TIGHT: the distinctive component phrase IMMEDIATELY
# adjacent (<=25 chars) to a value of the asked type. Loose windows would match the
# query terms that retrieval naturally surfaces (zeroing out all refusals); these only
# fire when the result literally states the answer (e.g. "banjo bolt: 25 Nm").
_REFUSAL_ASKS = (
    ("What's the exact tightening torque for the oil cooler line banjo bolt?",
     "oil cooler line banjo bolt tightening torque",
     r"banjo[^\n]{0,25}\d+\s*Nm|\d+\s*Nm[^\n]{0,15}banjo"),
    ("What's the fluid capacity for the angle gear (bevel gear)?",
     "angle gear bevel gear fluid capacity litres",
     r"(?:angle|bevel)\s*gear[^\n]{0,25}\d[\d.]*\s*(?:l\b|litre|liter)"),
    ("What wire color is the knock sensor signal at the ECM connector?",
     "knock sensor signal wire color ECM connector pin",
     r"knock\s*sensor[^\n]{0,25}(?:BN|BL|GN|RD|brown|blue|green|red|black|white|yellow)\b"),
    # NOTE: a part-number scenario ("secondary air injection check valve part number")
    # used to live here. Removed 2026-07-04: it taught "ambiguous part lookup -> refuse",
    # which directly contradicts build_cross_vehicle's "no exact match for this car, but
    # it shows up for a related vehicle -> answer with a verify-before-use caveat" — the
    # SAME trigger (retrieval didn't cleanly match), opposite lesson. That conflict was
    # confirmed to regress part_number accuracy 95%->70% in the 2026-07-03/04 retrains
    # (the model started refusing instead of using the cross-vehicle fallback). Keep
    # hard-refusal training scoped to torque/fluid/wire, where no competing pattern exists.
    ("What's the torque for the turbo oil return line banjo bolt?",
     "turbocharger oil return line banjo bolt torque",
     r"oil\s*return[^\n]{0,25}\d+\s*Nm"),
)


def build_refusal(doc: src.Doc) -> Optional[gc.Record]:
    """Pose a specific-value question whose answer the retrieved context does NOT
    contain (empty results), and target a refusal — the anti-hallucination example. We
    key it on the doc's vehicle so the model has a car but still no value. The question
    and the refusal wording both rotate (deterministically) so the model does not latch
    onto one literal string (see docs/AUDIT-2026-06-20.md)."""
    veh = src.pick_vehicle_for_doc(doc)
    if veh.is_empty():
        return None
    q, query, answerable = gc._pick(_REFUSAL_ASKS, doc.doc_id)
    split_key = f"{veh.model or '?'}|refusal"
    cid = gc.call_id(1)
    # The HARD refusal case the empty-result version never taught: real retrieval
    # returns RELATED-but-insufficient hits (these curated asks have no clean spec in
    # the corpus), and the model must still refuse instead of latching onto a nearby
    # number. Feed the real noisy list when available; empty list otherwise. This is
    # the live failure mode observed in the provenance test (non-empty results ->
    # the old model thrashed instead of refusing).
    live = _live_search(query, None)  # query-only, matching serve-time calls
    if live and re.search(answerable, json.dumps(live, ensure_ascii=False), re.I | re.S):
        # The corpus appears to actually answer this ask -> a refusal here would be a
        # contradictory signal (results contain the value, label says refuse). Skip.
        return None
    tool_payload: Any = live if live is not None else gc.search_docs_result([])
    msgs = [
        gc.system_msg(),
        gc.user_msg(q, vehicle=veh),
        gc.assistant_tool_call("search_docs", {"query": query}, cid),
        gc.tool_result(cid, tool_payload),
        gc.assistant_answer(gc.refusal_text(split_key + "|" + doc.doc_id)),
    ]
    return gc.make_record(msgs, example_type="refusal", answer_type="refusal",
                          vehicle=veh, split_key=split_key, source_ids=[])


# --------------------------------------------------------------------------- #
# 3. elicitation — vehicle-specific question, NO vehicle -> ask for it (D3)
# --------------------------------------------------------------------------- #
def build_elicitation(doc: src.Doc) -> Optional[gc.Record]:
    """A vehicle-specific question arrives with NO vehicle established. The model must
    ask for the VIN/engine/model+year (no tool call) — sticky-vehicle's front door."""
    topic = _topic_from_title(doc.title, "this part")
    q = f"What's the tightening torque for {topic}?"
    msgs = [
        gc.system_msg(),
        gc.user_msg(q),  # NO vehicle block
        gc.assistant_answer(gc.elicitation_text(doc.doc_id)),
    ]
    return gc.make_record(msgs, example_type="elicitation", answer_type="elicitation",
                          vehicle=None, split_key="none|elicitation", source_ids=[])


# --------------------------------------------------------------------------- #
# 4. cross_vehicle — this-car miss, surface an other-vehicle hit, flag verify (D4)
# --------------------------------------------------------------------------- #
def build_cross_vehicle(conn: sqlite3.Connection, part: src.Part) -> Optional[gc.Record]:
    """The active car has no match; the part appears for a RELATED vehicle. Answer
    with the applicability fallback + 'verify before use'. Never a supersession claim
    (that data doesn't exist, D4)."""
    profs = src.profiles_for_part(conn, part.part_number, limit=12)
    # D4: cross-vehicle must be HONEST. Find a related vehicle the part really serves
    # (carries an engine code), then a REAL sibling that shares that engine but whose
    # model the part does NOT list — a genuine this-car miss with a real shared basis.
    related = None
    for tup in profs:
        v = src.vehicle_from_profile(*tup)
        if v.engine and v.model:
            related = v
            break
    if related is None:
        return None
    part_models = {src._first_token(m) for (_e, m, _y) in profs if m}
    active = src.sibling_vehicle_sharing_engine(conn, related.engine, part_models)
    if active is None:
        return None  # no real engine-sharing sibling -> skip (never fabricate one)
    related_desc = (f"the {(related.year + ' ') if related.year else ''}{related.model} "
                    f"(shared engine {related.engine})").strip()
    q = f"Is there a part number for the {part.description.lower()} on my car?"
    cid = gc.call_id(1)
    msgs = [
        gc.system_msg(),
        gc.user_msg(q, vehicle=active),
        gc.assistant_tool_call("epc_part", {"component": part.description}, cid),
        gc.tool_result(cid, gc.epc_part_result([{
            "part_number": part.part_number, "description": part.description,
            "scope": "other vehicles", "applies_to": related_desc,
            "citation": f"EPC/PartItems/{part.part_number}",
        }], component=part.description)),
        gc.assistant_answer(gc.cross_vehicle_answer(
            f"The {part.description}", part.part_number,
            related_desc, confidence="medium")),
    ]
    return gc.make_record(msgs, example_type="cross_vehicle", answer_type="cross_vehicle",
                          vehicle=active, split_key=f"{active.model}|cross_vehicle",
                          source_ids=[part.part_number])


# --------------------------------------------------------------------------- #
# 5. multi_turn — sticky vehicle across turns; no topic bleed (D3)
# --------------------------------------------------------------------------- #
def build_multi_turn(doc_a: src.Doc, doc_b: src.Doc) -> Optional[gc.Record]:
    """Two turns, same vehicle, DIFFERENT topics. Turn 1 establishes the car + answers
    topic A. Turn 2 (no [VEHICLE] block — sticky) asks an unrelated topic B; the tool
    query is built from B's words ONLY (no A bleed)."""
    veh = src.pick_vehicle_for_doc(doc_a)
    if veh.is_empty() or doc_a.doc_id == doc_b.doc_id:
        return None
    topic_a = _topic_from_title(doc_a.title, "the first component")
    topic_b = _topic_from_title(doc_b.title, "the second component")
    c1, c2 = gc.call_id(1), gc.call_id(2)

    def _turn(doc: src.Doc, query: str):
        """Return (tool_payload, snippet, cite_ref, cite_id) grounded in real (or
        synthetic) retrieval, or None on a genuine miss."""
        results = _live_search(query, None)  # query-only, matching serve-time calls
        if results is not None:
            _, hit = _find_hit(results, doc)
            if hit is None:
                return None
            snip = " ".join(str(hit.get("snippet") or hit.get("lead") or "").split())[:160]
            if len(snip) < 30:
                return None
            return results, snip, (hit.get("vida_doc_ref") or doc.vida_doc_ref), (hit.get("doc_id") or doc.doc_id)
        snip = " ".join((doc.text_md or "").split())[:160]
        return (gc.search_docs_result([{"doc_id": doc.doc_id, "vida_doc_ref": doc.ref(),
                                        "title": doc.title, "snippet": snip, "scope": "this car"}]),
                snip, doc.vida_doc_ref, doc.doc_id)

    ta = _turn(doc_a, doc_a.title or topic_a)
    tb = _turn(doc_b, doc_b.title or topic_b)
    if ta is None or tb is None:
        return None
    payload_a, sa, ref_a, id_a = ta
    payload_b, sb, ref_b, id_b = tb
    msgs = [
        gc.system_msg(),
        gc.user_msg(f"How do I service {topic_a}?", vehicle=veh),
        gc.assistant_tool_call("search_docs", {"query": doc_a.title or topic_a}, c1),
        gc.tool_result(c1, payload_a),
        gc.assistant_answer(f"{sa[:140]} {gc.citation(ref_a, id_a)}"),
        # turn 2: same car (no [VEHICLE] block), unrelated topic, fresh query
        gc.user_msg(f"And what about {topic_b}?"),
        gc.assistant_tool_call("search_docs", {"query": doc_b.title or topic_b}, c2),
        gc.tool_result(c2, payload_b),
        # No "for the <engine>" here: the vehicle is sticky from turn 1, NOT present in
        # turn 2's tool result, so re-asserting it would be an ungrounded fact. Cite the
        # retrieved source only (consistent with build_grounded_procedure).
        gc.assistant_answer(f"{sb[:140]} {gc.citation(ref_b, id_b)}."),
    ]
    return gc.make_record(msgs, example_type="multi_turn", answer_type="procedure",
                          vehicle=veh, split_key=f"{veh.model or '?'}|multiturn",
                          source_ids=[doc_a.ref(), doc_b.ref()])


# --------------------------------------------------------------------------- #
# 6. figure — figure-retrieval only; no callout->part (D5)
# --------------------------------------------------------------------------- #
def build_figure(doc: src.Doc) -> Optional[gc.Record]:
    figs = doc.figure_ids()
    if not figs:
        return None
    veh = src.pick_vehicle_for_doc(doc)
    topic = _topic_from_title(doc.title, "this component")
    cid = gc.call_id(1)
    msgs = [
        gc.system_msg(),
        gc.user_msg(f"Show me the figure for {topic}.", vehicle=veh if not veh.is_empty() else None),
        gc.assistant_tool_call("get_figures", {"query": doc.title or topic}, cid),
        gc.tool_result(cid, gc.get_figures_result(
            doc_id=doc.doc_id, title=doc.title, figures=figs[:4],
            scope="this car", vida_doc_ref=doc.ref())),
        gc.assistant_answer(gc.figure_answer(doc.ref(), len(figs[:4]), topic)),
    ]
    return gc.make_record(msgs, example_type="figure", answer_type="figure",
                          vehicle=veh if not veh.is_empty() else None,
                          split_key=f"{veh.model or 'any'}|figure", source_ids=[doc.ref()])


if __name__ == "__main__":
    import os

    # Pure-builder self-check on synthetic rows (no DB needed) — every builder's
    # output must satisfy the contract validator.
    doc = src.Doc(doc_id="DiagSwdl:Document:-112365", vida_doc_ref="VCC-112365-1",
                  title="Replacing the thermostat", model="XC70", year="2004",
                  engine="B5254T2", section="Repair",
                  text_md="| Intake manifold bolts | 19 Nm |\nTighten in sequence.",
                  image_refs='["0900c8af8005dd79_0_0.gif"]')
    doc2 = src.Doc(doc_id="DiagSwdl:Document:-99999", vida_doc_ref="VCC-137506-1",
                   title="Replacing the coolant pump", model="XC70", year="2004",
                   engine="B5254T2", section="Repair",
                   text_md=("Drain the coolant into a clean container. Remove the "
                            "auxiliary belt and the pump pulley. Unbolt the coolant "
                            "pump from the block and discard the gasket. Fit a new "
                            "gasket, install the pump and torque the bolts to 24 Nm."),
                   image_refs=None)
    part = src.Part(part_number="1271998", description="Heated oxygen sensor",
                    section_id=1, section_title="Fuel system")

    builders = {
        "grounded_torque": build_grounded_torque(doc),
        "grounded_procedure": build_grounded_procedure(doc2),
        "elicitation": build_elicitation(doc),
        "refusal": build_refusal(doc),
        "figure": build_figure(doc),
        "multi_turn": build_multi_turn(doc, doc2),
    }
    import json as _json
    for name, rec in builders.items():
        assert rec is not None, f"{name} returned None"
        problems = gc.validate_messages(rec.messages)
        assert not problems, f"{name}: {problems}"
        print(f"  [ok] {name:20} types={[m['role'] for m in rec.messages]}")

    # train==serve envelope conformance: the tool-result content must match the live
    # server's shape (search_docs/lookup_pin -> list; epc_part -> {"rows"}; get_figures
    # -> dict). Parse each tool message's JSON content and assert the top-level shape.
    def _tool_contents(rec):
        return [_json.loads(m["content"]) for m in rec.messages if m.get("role") == "tool"]
    assert isinstance(_tool_contents(builders["grounded_torque"])[0], list), "search_docs must be a bare list"
    assert isinstance(_tool_contents(builders["multi_turn"])[0], list)
    fig = _tool_contents(builders["figure"])[0]
    assert isinstance(fig, dict) and fig.get("doc_id"), "get_figures must be a summary dict with doc_id"
    # verbatim grounding: the torque value the model asserts must ALSO be visible in the
    # tool-result the model READS (not just the answer) — else it's memorized, not grounded.
    tq = first_torque(doc.text_md or "")
    assert f"{tq.value_nm:g} {tq.unit}" in builders["grounded_torque"].messages[-1]["content"]
    assert f"{tq.value_nm:g}" in builders["grounded_torque"].messages[3]["content"], \
        "torque value not visible in the tool-result the model reads (grounding gap)"
    assert _value_visible(_tool_contents(builders["grounded_torque"])[0], tq.value_nm), \
        "_value_visible disagrees with the grounded torque tool result"
    # _value_visible must reject a number that's only a part-number tail / decimal frac,
    # and accept a genuine torque spec (regression guard for the lookbehind).
    assert not _value_visible([{"snippet": "PN400 Nm"}], 400.0), "matched a part-number prefix as torque"
    assert not _value_visible([{"snippet": "ratio 1.400 Nm"}], 400.0), "matched a decimal fraction as torque"
    assert _value_visible([{"snippet": "Tightening torque: 400 Nm"}], 400.0), "missed a real torque value"
    # refusal/elicitation variants render (no {"results"} leakage anywhere)
    for rec in builders.values():
        for m in rec.messages:
            if m.get("role") == "tool":
                assert '"results"' not in m["content"], "stale {'results':...} envelope present"
    print("examples.py self-check OK (builders produce contract-valid, server-shaped records)")
