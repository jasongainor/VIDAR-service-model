# BUILD_CONTRACT.md — interfaces every module conforms to

This pins the seams so independently-built modules cohere. **Do not invent shapes
that contradict this file.** The load-bearing foundation modules already exist and
are correct; build the rest against them.

## Conventions
- Python 3.12. **Mac-side code (`grounding/`, `dataset/`, `vida-eval/`) is stdlib-only**
  (sqlite3, json, re, dataclasses, urllib) — it must run on the Mac with NO new deps.
  The eval *judge* and the *training* code may import third-party packages, but those
  imports must be **lazy / inside functions** so the module still `py_compile`s and the
  deterministic paths still import on a bare Mac.
- **Never put torch/unsloth/transformers in the Mac repo's pyproject.** Training deps
  live in `requirements-train.txt` (Ubuntu box only).
- Every `.py` must `python -m py_compile` clean. Add a `if __name__ == "__main__":`
  self-check to anything runnable.
- Reference files as real paths; match the surrounding code's style.

## grounding contract (import, don't reimplement)
`from grounding import contract as gc` (or `from grounding import ...`). Key surface:
- `gc.SYSTEM_PROMPT`, `gc.TOOLS` (OpenAI function specs mirroring the MCP server), `gc.TOOL_NAMES`
- `gc.Vehicle(year, model, engine, vin)` → `.label()`, `.vehicle_block()`, `.is_empty()`, `.to_meta()`
- builders: `gc.system_msg()`, `gc.user_msg(text, vehicle=None)`, `gc.assistant_tool_call(name, args:dict, call_id)`,
  `gc.tool_result(call_id, content)`, `gc.assistant_answer(text)`, `gc.call_id(n)`
- `gc.citation(vida_doc_ref=None, doc_id=None)` → `"(VCC-…)"`; templates `gc.REFUSAL_TEMPLATE`,
  `gc.ELICITATION_TEMPLATE`, `gc.cross_vehicle_answer(...)`, `gc.figure_answer(...)`
- `gc.make_record(messages, example_type=, answer_type=, vehicle=, split_key=, source_ids=)` → `Record`
- `gc.validate_messages(messages)` → list of problems (empty == valid). **Every generated
  record and benchmark case MUST pass this.**

### Canonical JSONL training record (one per line)
```json
{"messages":[
  {"role":"system","content":"<gc.SYSTEM_PROMPT>"},
  {"role":"user","content":"[VEHICLE] 2004 XC70 | B5254T2 [/VEHICLE]\nTorque for the intake manifold?"},
  {"role":"assistant","content":null,"tool_calls":[{"id":"call_0001","type":"function",
     "function":{"name":"search_docs","arguments":"{\"query\":\"intake manifold tightening torque\"}"}}]},
  {"role":"tool","tool_call_id":"call_0001","content":"[{\"doc_id\":\"…\",\"vida_doc_ref\":\"VCC-112365-1\",\"snippet\":\"Tightening torque: 15 Nm\",\"scope\":\"this car\"}]"},
  {"role":"assistant","content":"15 Nm (VCC-112365-1), for the B5254T2."}
],"metadata":{"example_type":"grounded","answer_type":"torque","vehicle":{...},"split_key":"…","source_ids":["VCC-112365-1"]}}
```
Example types + target mix live in `gc.EXAMPLE_TYPES`. Citations use real
`vida_doc_ref` (preferred) or `doc_id` — **never invent ids**.

**Tool-result shapes MIRROR the live MCP server** (train==serve, D17): `search_docs` /
`lookup_pin` → a **bare JSON list**; `epc_part` → `{"query":…,"rows":[…]}`; `get_figures`
→ a summary dict. Build them via `gc.search_docs_result(...)` / `gc.lookup_pin_result(...)`
/ `gc.epc_part_result(...)` / `gc.get_figures_result(...)` — NEVER the old
`{"results":[…]}` wrapper (a shape the server never emits).

## eval scorecard (metrics.py defines; score.py emits) — JSON shape
```json
{"model":"qwen3-4b","config":"r16","dataset_version":"v1","n":574,
 "metrics":{
   "part_number_exact":{"pass":.., "n":.., "rate":0.0},
   "torque_exact":{"rate":0.0},
   "procedure_correct":{"rate":0.0},        // LLM-judge
   "retrieval_hit_at_k":{"k":8,"rate":0.0}, // gate
   "hallucination_rate":0.0,                // answered when should refuse
   "refusal_correct":{"rate":0.0},
   "figure_correct":{"rate":0.0}},
 "gates":{"part_number_exact":0.995,"torque_exact":0.995,"hallucination_rate":0.005,"refusal_correct":0.98},
 "passed": false}
```
Deterministic metrics (part#/torque exact, refusal detection, hit@k) computed in code
($0). `procedure_correct` + hallucination judgment via `judge.py` (Claude API, Sonnet
4.6 via Batches; Opus only for champion tiebreak). Reuse `vida-eval/retrieval.py`'s
in-process `search_docs` import for hit@k.

## VIDA source DB schema (for dataset/ + benchmark_gen.py) — data/vida-kb.sqlite3
- `documents(doc_id PK, db, table_name, source_pk, vida_doc_ref, title, profile[JSON],
   model, year, engine, ecu, section, text_md, image_refs[JSON])` — 124,040 rows.
  Torque lives in `text_md` (spec tables like `| Oil drain plug | 40 Nm |` AND inline
  prose `tighten to 40 Nm`). ~125 docs have title `Tightening torque`. Use `torque.py`.
- `epc_parts(part_number, description, function_group, section_id, section_title, type_id)`
  — 1.08M rows, 150k DISTINCT part_numbers. `part_number→description` unambiguous;
  reverse is one-to-many (43k 'Flange screw'). EXCLUDE part_number='ns' / blank desc.
- `epc_section_profiles(section_id, profile_id)` 477k — part→section→profiles (cross-vehicle).
- `doc_profiles(doc_id, profile_id)` 2.6M — doc→vehicle.
- `profiles(profile_id, description, model, year, engine, folder_level)` 139k — the vehicle lens.
- `pinouts(...)` 433 (one vehicle, volvo_s60r_2005). `components` 396.
- `data/vida-images.sqlite3`: `images(path PK, content_type, data BLOB)` 71k;
  `graphics(title, path)` 71k. Figures resolve via documents.image_refs → graphics/images.
- A doc's `vida_doc_ref` (e.g. VCC-112365-1) is the citation token; `doc_id` is the fallback.
- **Cross-vehicle = fallback** and **figures = retrieval only** (no callout→part) per DECISIONS.
- Split key for the frozen benchmark: `f"{model}|{topic}"` — split by vehicle AND topic,
  never random (VIDA rows are duplicated across platform-mates).
