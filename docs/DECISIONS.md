# DECISIONS.md

Append-only. Each decision: what, why, and what it rules out. Seeded from the
2026-06-20 plan validation + design pass.

## D1 — Train shape: function-calling agent (not flat `[CONTEXT]`)
**Decided 2026-06-20.** The model is trained on the real message sequence it is
served under: system prompt → user → assistant `tool_calls` → `role:"tool"` JSON →
assistant grounded answer. **Why:** the live system is native MCP function-calling
(`src/server.py` + Open WebUI external tool server), not a pipeline that pre-injects a
`[CONTEXT]` block. Training on inlined context would teach inputs the model never sees
at serving time. The validation also showed the real failure mode is small models
*thrashing in tool-call loops* (the 60-turn runaway, `cam-seals-compound`) — disciplined
tool use is the actual objective. **Rules out:** a `[CONTEXT]`-reader dataset and a
new Open-WebUI rewriting pipeline.

## D2 — One shared contract (`grounding/contract.py`)
Imported by the dataset generator, the eval harness, and the serving system prompt, so
train == serve by construction. Changing it is a versioned act (`CONTRACT_VERSION`).

## D3 — Sticky vehicle + elicit-on-missing; topic stays independent
**Decided 2026-06-20 (refines [[local-llm-multiturn-lessons]]).** The vehicle persists
across a conversation (turn 1 establishes it via VIN/engine/model+year or a `[VEHICLE]`
block; later turns inherit it). The model asks for the vehicle when a vehicle-specific
question arrives without one, and works without one for vehicle-agnostic questions.
**But** the *topic/component* still must not bleed across turns (the existing anti-drift
guard cases). This **reverses** the current "throw context out every turn" system prompt
for the *vehicle* only. The contract + guard cases must test both (sticky vehicle AND
no topic contamination). **Why:** user wants a conversational bot that holds the car and
prompts for it, not one that re-asks every turn.

## D4 — Cross-vehicle is a fallback, not a default
With a vehicle in context, answer for that car only; surface other-vehicle results
**only** on a this-car miss or when the stored car description is too weak. No-car
context → applicability-only. **Why:** user does not want to be "spammed with other
cars". Maps onto the existing this-car/other-vehicles tiering. Supersession chains are
NOT in the data → cross-vehicle is *applicability*, never a supersession claim.

## D5 — Figures: retrieval only (no callout→part)
"Show me the figure for X" ships (images resolve 100% to baked bytes); the
callout-number→part-number sentence is cut for v1. **Why:** that mapping does not exist
in the data (exploded-view BOMs are external GR-##### WebCGM, balloon↔part is render-time
only). Generating it would be fabrication — against the user's standing no-fake-numbers
rule. Reallocated the freed share to part#/torque/pinout.

## D6 — Torque needs a parser with a measured precision number
`LIKE '%Nm%'` (7,224 docs) is polluted; real value signal ~2,198 docs, ~125 clean spec
tables (some malformed). `dataset/torque.py` parses spec tables + inline prose, discards
"see VCC-…" stubs, and reports measured precision. **Rules out:** claiming "100%
deterministic torque".

## D7 — Citation token = real id (`vida_doc_ref` → `doc_id`)
The model cites `VCC-118802-1` (or `doc_id`), resolvable by eval against
`documents.vida_doc_ref`. **Rules out:** invented `proc-21-345`-style ids.

## D9 — Coordinator owns the reserve/release lifecycle
`reserve(host)` happens on the coordinator before an exclusive (train) lease is handed
out; `release(host)` on done/failed/interrupted AND in the reaper. **Why:** a dead
worker still gets its GPU returned to inference — no reserve leak.

## D10 — Capability-based workers; the 3080 is never idle
Hosts advertise a capability manifest, not a fixed role. A host is `reserve`d only when a
leased job genuinely needs exclusive VRAM; otherwise the router keeps it serving
inference/eval in the gap. The 3080 takes 3–4B train jobs AND eval. **Why:** user: "why
not both? I don't want it idle because it is gated to a job."

## D12 — Eval judge = Claude API, bounded and offline
Sonnet 4.6 via the Batches API (50% off, not latency-sensitive); Opus 4.8 only to break a
tie on the champion. Candidate **answers are generated locally** (free); ~60% of the
benchmark (part#/torque exact, refusal detection, hit@k) is graded deterministically
($0); Claude judges only procedure-correctness + hallucination. Estimated $4 (100x
sanity) to $14–47 (rigorous). **Never** use Claude to generate the dataset or serve the
bot (that is the only "mortgage payment" risk). **Why:** user is cost-anxious; see the
math in `vida-eval/judge.py` header.

## D13 — Disk
Checkpoint store capped at **50 GB** on the coordinator with oldest-non-champion
eviction.

## D15 — Validate-first format smoke-test
Before any multi-hour run, a 50-example format smoke-train + load on EVERY candidate
(P5 smoke campaign). Cheap insurance against "won't work on this model, redo in two
weeks". Encoded as a campaign, gated by `grounding.contract.validate_messages`.

## D16 — Reuse `vida-eval/`, don't fork `/eval`
The fine-tune eval extends the existing `vida-eval/` scorecard so the model is judged by
the same instrument the retrieval lever is already guarded by.

## D19 — Served model = P2-default "take a stab", not elicit-on-missing (2026-07-01)
The TRAINING contract (D-locked "sticky vehicle + elicit-on-missing", `SYSTEM_PROMPT`) makes the
model ask for the vehicle when none is given. In real single-operator use that manifested as the
model **opening the first turn by demanding a VIN/FIN** — which is a dead end: there is **no VIN
decoder** in `src/server.py` (nor in the model), so nothing can turn a VIN into a scope, and the
turn just stalls. Meanwhile the retrieval layer already assumes this build's car — `search_docs`
defaults to the baked-in 2005 S60R (B5254T4) and blends shared **P2-platform** (S60/S80/V70/XC70/
XC90) results ([src/server.py:733](../src/server.py#L733)). So the model *had* everything it
needed to answer and asked anyway.

Decision: the **served** prompt gets a P2-default "take a stab" profile
(`grounding/contract.SERVING_SYSTEM_PROMPT` / `serving_system_msg()`, used by
`scripts/cloud/install_owui_model.py` and mirrored into the Open WebUI model row's `params.system`).
It (1) tells the model it works on Volvo P2 cars and to ASSUME the platform unless told otherwise;
(2) on a cold vehicle-specific turn, **searches first and answers with the citation, stating which
engine the source applies to** (e.g. "25 Nm (VCC-136956-1), for the B5254T4") — the user reads the
applicability and can correct it; (3) **never asks for a VIN/FIN** — the retrieval-miss fallback
asks for engine code or model+year (the things that actually scope retrieval). Every safety rule is
unchanged: retrieve-or-refuse, cite the source, cross-vehicle flagging, no topic bleed, never invent
(verified: a "front strut top mount" query still refuses rather than grab the nearby roll-link 50 Nm).

This is a **serving-time behavioral delta from the training contract** (the model was trained with
elicit-on-missing; the prompt suppresses that in favor of the already-dominant grounded behavior — a
strong steer a 7B instruct model follows). Reconcile on the next training run by regenerating the
`elicitation` examples (10% share) as P2-default take-a-stab + narrow-by-engine-code, restoring
train==serve. Benchmark note: the frozen `vida-eval` still scores the OLD elicitation cases, so a
served model on this profile will "miss" those — expected and accepted; they measure the retired
behavior.
