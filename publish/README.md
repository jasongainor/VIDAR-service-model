---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
pipeline_tag: text-generation
tags: [qwen2.5, lora, qlora, volvo, vida, automotive, grounded-qa, function-calling, gguf]
---

# VIDAR-Grounded-7B

A **Qwen2.5-7B-Instruct** fine-tune for **grounded Volvo VIDA service Q&A**. It answers automotive service questions (torque specs, pinouts, part numbers, procedures) as a tool-calling agent that **cites its source documents** (`VCC-…` refs) and **refuses rather than fabricate** when it lacks a grounded source.

> **⚙️ Setup harness:** [**github.com/jasongainor/VIDAR-service-model**](https://github.com/jasongainor/VIDAR-service-model) — a zero-input repo that builds the retrieval layer from *your own* licensed VIDA install and wires this model into Open WebUI. Hand it to a coding agent and it does the whole setup.

## Evaluation
Winner of a 9-way shootout (Qwen2.5-7B / Qwen3-4B / Llama-3.1-8B / Llama-3.2-3B across LoRA ranks), scored on a curated VIDA benchmark through the real retrieval + grader path:

| Metric | Score |
|---|---|
| Type-correct | **91.7%** (11/12) |
| Honest-citation | 100% |
| Retrieval-hit | 100% |
| Grounded | 100% |
| Errors | 0 |

## Intended use
Grounded Volvo (VIDA) service QA, used **with a retrieval/tool layer** (MCP). It cites source docs and refuses on missing/ambiguous grounding. Not a general assistant.

## Training
- **Method:** QLoRA (r=16, alpha=32), 2 epochs, max_seq 8192, lr 2e-4 (Unsloth).
- **Response-only masking:** trains only on assistant turns (tool_calls + grounded answer), never on the system prompt or tool-result JSON — so it learns to ground, not to invent facts.
- **Base:** [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Apache-2.0).

## Prompt format
Qwen2.5 chat template with tools (function calling): `system → user → assistant(tool_calls) → tool → grounded answer`.

## Limitations
- Domain-specific to Volvo VIDA; requires the retrieval/tool layer for grounding — standalone it will refuse or under-cite.
- Coverage limited to the VIDA extract it was trained on.

## How to use

This is a **tool-grounded** model: it answers from retrieval tools, not memory. You need three things — the model (this GGUF), the **system prompt** below, and a **retrieval layer** exposing 5 tools over your VIDA data.

### 1. Run the model (Ollama)
```bash
# Download VIDAR-Grounded-7B-Q4_K_M.gguf from this repo, then:
cat > Modelfile <<'EOF'
FROM ./VIDAR-Grounded-7B-Q4_K_M.gguf
PARAMETER temperature 0.1
PARAMETER num_ctx 8192
EOF
ollama create vidar-grounded-7b -f Modelfile
```
(Leave the chat template to Ollama's GGUF-inferred default — it round-trips Qwen2.5 tool-calls.)

### 2. System prompt (required)
The model was trained to obey this exact grounding contract. Set it as your system prompt:
```
You are a Volvo VIDA repair assistant. You answer ONLY from the VIDA knowledge-base tools; you never answer car facts from memory. A wrong torque, wire color, fluid capacity, or part number can damage a car or injure someone — "I could not verify that" is a correct answer, inventing a value is a failure.

TOOLS (call them; do not guess):
- search_docs(query, car?) — hybrid search over procedures/specs/wiring. Returns ranked snippets with a source id and a scope ('this car' | 'other vehicles'). Open the full text with get_document before quoting a value.
- get_document(doc_id) — full text of one document.
- epc_part(part_number? , component?) — EXACT parts catalogue lookup. Use this (NOT search_docs) when given a specific part number, or asked for the part number(s) of a named component (oxygen sensor, thermostat, ...). Returns description + applicability scope.
- lookup_pin(module?, pin?, query?) — wiring / connector pinouts.
- get_figures(query?, doc_id?) — relevant figure(s) for a procedure. You can show a figure; you CANNOT name a callout->part number (that data does not exist) — describe the figure and cite its document instead.

VEHICLE CONTEXT (sticky):
- The vehicle is whatever the user established earlier in THIS conversation (a VIN, engine code, or model+year), or a [VEHICLE]...[/VEHICLE] block in their message. Once set, it carries to every later turn — "what about the exhaust manifold?" reuses the same car. Do not ask again once you have it.
- If a vehicle-specific question arrives and NO vehicle is established, ask for the VIN, engine code, or model and year before answering — do not guess a vehicle. Answer vehicle-agnostic questions directly without asking.
- Pass car=<engine code> to search_docs only to deliberately retarget another vehicle.

NO TOPIC BLEED: a new question is a NEW topic unless it explicitly refers back ("it", "that one", "while I'm in there"). Build the tool query from the latest question's words only; never mix in a component or system from an earlier turn. The VEHICLE persists across turns; the TOPIC does not.

ANSWERING:
- Lead with the answer, then cite the source id(s) in parentheses, then state scope if it is not this car. Example: "15 Nm (VCC-112365-1), for the B5254T2."
- For part numbers, route to epc_part and prefer 'this car' parts; say so.
- CROSS-VEHICLE IS A FALLBACK: with a vehicle in context, answer for that car and do not volunteer other vehicles. Only if this car yields no answer (or its description is too weak to be sure) may you surface an other-vehicle result — and then flag it: "appears for a related vehicle — verify before use".
- If the tools return nothing that answers the question, refuse: say you could not find a verified value and ask for the VIN / engine code / model+year to narrow it.
- If sources conflict on a value, present the conflict and prefer this car's source; do not silently pick one.
```

### 3. The 5 tools (implement against YOUR VIDA data)
The model emits tool-calls for these; your retrieval layer must answer them:
- `search_docs(query, car?)` — hybrid search over procedures/specs/wiring → ranked snippets w/ source id + scope.
- `get_document(doc_id)` — full text of one document (by doc_id or VCC ref).
- `epc_part(part_number?, component?)` — exact parts-catalogue lookup.
- `lookup_pin(module?, pin?, query?)` — wiring / connector pinouts.
- `get_figures(query?, doc_id?)` — figures for a procedure.

### 4. Connect it to Open WebUI
Run the reference MCP server over HTTP and point Open WebUI (via `mcpo`) at it:
```bash
uv run python src/server.py --http 8000        # streamable HTTP MCP server
```
Then add the model in Open WebUI (Ollama backend), paste the system prompt above, and register the MCP tool endpoint. Ask a vehicle-specific question and it will call the tools.

### 5. Bring your own VIDA data
The reference tools read a single SQLite store (`data/vida-kb.sqlite3`) built from a VIDA extract. The easiest path is the companion harness — [**VIDAR-service-model**](https://github.com/jasongainor/VIDAR-service-model) — which reads your install's own SQL credentials, runs the whole extract/build pipeline (`build_db → extract_docs → extract_epc → build_synonyms → extract_images → build_embeddings`), serves the MCP endpoint, and wires Open WebUI — hands-free. **No VIDA data is distributed with this model** — you supply your own from a licensed install; the model only provides the grounded reasoning + citation behavior.

### Example (Open WebUI)
> **User:** `[VEHICLE] 2004 S40 (04-) | B4164T3 [/VEHICLE]` What's the tightening torque for the front brake caliper?
>
> **Assistant:** *(calls `search_docs`)* → **120 Nm (VCC-370744-1), for the B4164T3.**

Identify the car by **model + year + engine code** (the model does **not** decode VINs). Vehicle-agnostic questions are answered directly; a vehicle-specific question with no car set triggers an elicitation for the engine code / model+year.

## License
Apache-2.0, inherited from the Qwen2.5-7B-Instruct base.
