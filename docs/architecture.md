# VIDA-KB — architecture & distribution roadmap

This is the "how does someone else run this" document: the consumability/distribution
design — ship CODE, build DATA locally from your own VIDA license.

## The shape: ship CODE, build DATA locally

VIDA content is Volvo's proprietary/licensed material. The repo therefore ships
**only code + docs** and points at the **user's own VIDA install** (bring-your-own
licensed data, like an emulator needing your own BIOS). Nothing VIDA-derived is ever
committed — not the SQLite store, not `mssql-data/`, not extracted images/text, not
embeddings. `git ls-files` must be audited for proprietary content before any publish.

This forces a clean three-layer split, run in order:

```
  clone repo → configure (car.toml, .env) → attach your VIDA DBs to vida-sql
     │
     ├─ LAYER 1  INGESTION  (one-time, offline, setup-heavy)
     │     build_db.py + extract_docs.py + extract_epc.py + attach_dbs.sql
     │     → produces the LOCAL ARTIFACT (data/vida-kb.sqlite3, + future: images, vectors)
     │
     ├─ LAYER 2  SERVER  (src/server.py — the shipped runtime)
     │     reads the artifact, exposes MCP tools, enforces vehicle/engine applicability
     │
     └─ LAYER 3  CLIENT  (Open WebUI / Claude Desktop / any MCP client)
           points at the server; renders citations and figures to the user
```

So **the server is inert until you have ingested your own data.** The server is a
constant (ships ready); the artifact is the per-user product of ingestion. "Built after
ingestion" is the right mental model — not compiled per-user, but **non-functional
without the locally-built store.**

## Current runtime dependencies (the consumability gap)

**Self-contained as of 2026-06-16** — the live SQL Server is now a SETUP-time dependency
only; every tool reads local artifacts at runtime (live SQL remains a transparent
fallback for anything not yet baked):

| Tool | Reads from | Runtime dep |
|------|-----------|-------------|
| `search_docs`, `get_document`, `lookup_pin`, `list_sources` | `data/vida-kb.sqlite3` (FTS5 + synonyms) | artifact only ✅ |
| `search_docs` (semantic recall) | `data/vida-vectors.sqlite3` (sqlite-vec) + local embed endpoint | local ✅ |
| `epc_part` | `epc_parts` + `epc_section_profiles` in the store (built by `extract_epc.py`) | artifact only ✅ |
| `get_figures` / image server (bytes) | `data/vida-images.sqlite3` (baked blobs) | artifact only ✅ (live fallback) |

Local artifacts and how they're built:
- **`data/vida-kb.sqlite3`** — `build_db.py` (schema + pinouts) → `extract_docs.py` (docs) →
  `extract_epc.py` (EPC docs **and** the `epc_parts`/`epc_section_profiles` lookup tables) →
  `build_synonyms.py` (the `synonyms` table mined from the en-US/en-GB Lexicon).
- **`data/vida-images.sqlite3`** — `extract_images.py` bakes the ~71k referenced raster
  bytes (~2 GB) out of the live `imagerepository`. WAL mode, so the running server reads
  it while a re-extraction writes.
- **`data/vida-vectors.sqlite3`** — `build_embeddings.py` embeds every doc with a local
  model (nomic-embed via LM Studio/Ollama) into a sqlite-vec store (768-dim).

Only the *referenced* images are baked (the doc corpus is ~all raster, ~100% resolvable);
CGM vector diagrams stay deferred. `extract_images.py` also bakes a `graphics(title, path)`
map (71,021 rows) into `vida-images.sqlite3`, so the marker→path resolution for non-1:1 docs
(the ~25% whose markers aren't 1:1 with `image_refs`) is local too — `_ordered_paths` reads
the baked map first, live `LocalizedGraphics` only as a fallback. So `get_figures` needs **no
live SQL at all** (verified: figures resolve and serve with the SQL Server offline). CGM
remains the only image gap.

Only *referenced* images need baking (the doc corpus is ~all raster, ~100% resolvable —
see `images-plan.md`), not all 92k graphics. CGM vector diagrams stay deferred.

## Vehicle context (done — 2026-06-16)

All car-specific config lives in one `CarContext` (`src/server.py`), resolved once at
import: the example default vehicle ← optional `car.toml` ← `VIDA_CAR_*` env vars. Retrieval
scoping, EPC applicability, the engine-family safety gates and every user-/model-facing
label (incl. tool docstrings, via the `_tool()` localizing decorator) read from it. So a
different VIDA install retargets the tool with a **one-file edit**, no code changes. See
`car.example.toml`.

This is per-**deployment** (one car per running instance — matches the real use case: you
own one car). A later step can make the context a per-**query** parameter for the
all-cars goal; the `CarContext` object is the seam that makes that a clean extension
rather than a rewrite.

## Engine-applicability enforcement (done — the safety primitive)

The non-negotiable invariant, enforced — not prompted: retrieval must **never cross
engine families** (a 6-cyl diesel head-bolt torque is dangerous on a 5-cyl petrol). One
shared classifier (`_engine_applicability`) + a content backstop (`_lead_wrong_engine`)
gate every read path — `search_docs` withholds wrong-engine lead text, `get_figures`
drops wrong-engine figures and refuses rather than substitute, `get_document` warns. This
layer sits **on top of whatever retrieval produces the candidates**, so it survives the
embedding upgrade below unchanged. Guard: `tests/test_engine_scope.py`.

## Embedding / hybrid retrieval (planned — the "correct over speed" lever)

Today retrieval is lexical only (FTS5/BM25 + synonym expansion). That misses
semantically-phrased queries ("make the AWD quieter" → angle-gear/propshaft docs) and
mis-ranks when discriminating terms are rare. Embeddings are the parked retrieval-ranking
lever. They thread cleanly through the layering — **produced at ingestion, consumed at
retrieval:**

- **Layer 1 (ingestion):** chunk + embed each doc with a **local, offline** embedding
  model → store vectors in the artifact (e.g. `sqlite-vec`, keeping it one file). Vectors
  are VIDA-derived → **local-only, never shipped** (same legal line as the store).
- **Layer 2 (server):** retrieval becomes **hybrid** — keep FTS5/BM25 for exact tokens
  (part numbers, torque values, VCC refs, where lexical is *better*) and fuse with vector
  search for semantic recall. The applicability/engine gates are unchanged and still run
  on the fused candidate set.

### Synonym sourcing — stop hand-curating `_SYN`

Hand-maintaining the synonym map is unwinnable whack-a-mole (the "pattern"≠"sequence" miss
is one of endless permutations). Two replacements, in priority:

1. **Mine VIDA's own EPC `Lexicon`** (offline, domain-exact, ship-safe). The same
   `DescriptionId` carries en-US (`fkLanguage=15`) and en-GB (`16`) wording — a built-in
   mechanic-speak→VIDA-wording dictionary. A live count found **3,254 differing US/UK
   pairs** (headlight→headlamp, A/B/C-post→pillar, ABS module→unit, air cleaner
   housing→filter housing). At ingestion, clean these (drop `Gcp` internal markers,
   `L.H./R.H.`, typos, punctuation-only diffs) into the synonym layer. Ship-safe because
   the *code* derives it from the user's own data; the terms are never committed.
2. **Embeddings make most synonyms moot** — semantic similarity covers boot≈trunk,
   whine≈noise, pattern≈sequence with zero enumeration. `_SYN` then shrinks to exact-match
   aids only (abbreviations, part-number formats).

External lexicons (PCdb/ACES-PIES from the Auto Care Association; WordNet/ConceptNet) exist
but are noisier and US-catalog-flavored; VIDA-Lexicon + embeddings dominates them here.

**Built 2026-06-16** (`build_embeddings.py` + `_augment_with_vectors` in the server):
1. **Embedding model** — `nomic-embed-text-v1.5` (768-dim) via the local LM Studio /
   Ollama OpenAI-compatible endpoint (`VIDA_EMBED_URL` / `VIDA_EMBED_MODEL`); ~157/s,
   whole corpus in minutes. nomic task prefixes used (`search_document:` / `search_query:`).
2. **Chunking** — whole-doc for v1 (title + first 6k chars; VIDA docs are short).
3. **Fusion** — **conservative additive**: FTS ranking is preserved exactly; vector-only
   hits fill *empty* result slots (so a strong exact match is never displaced). A full
   weighted RRF re-rank is deferred — naive equal-weight RRF was observed to demote good
   exact matches, so weight tuning waits for the `vida-eval` set. Engine/scope enforcement
   runs on the fused set, unchanged.

## Roadmap (sequencing)

1. ~~Self-contained artifact~~ — **DONE** (EPC + images baked; live SQL is setup-only).
2. ~~Embeddings / hybrid retrieval~~ — **DONE** (vector recall + weighted RRF).
3. ~~Tune retrieval~~ — **DONE 2026-06-16** (see "Retrieval tuning" below): a deterministic
   `vida-eval/retrieval.py` scorecard, weighted-RRF fusion (low-confidence regime), lexical
   mis-rank fix (spec-table + title boost + bolt↔screw synonym), embedding precision gate,
   and cross-vehicle torque-**conflict flagging** (the M6 19-vs-20 Nm case).
4. ~~Fully sever live SQL~~ — **DONE** (figure title→path `graphics` map baked; `get_figures`
   needs no live `LocalizedGraphics`, verified with the SQL Server offline).
5. ~~Per-query car context / all-cars~~ — **DONE 2026-06-16**: `search_docs(car='<engine>')`
   retargets tiering AND the engine-applicability safety gate to a different vehicle in one
   process (`_resolve_car`); ingestion was already de-filtered (the store tags all profiles).
   Still per-deployment for `epc_part`/`lookup_pin` (their own profile resolution is the
   remaining all-cars work).
6. ~~CGM Milestone 2~~ — **mechanism DONE 2026-06-16**: `extract_cgm.py` renders the few
   document-referenced CGMs to SVG via an external converter (jcgm-style, `VIDA_CGM_CMD`) and
   bakes them into the image store; `get_figures` + the image server serve baked SVGs. The
   actual render needs a JDK + jcgm (see `images-plan.md`); absent one it degrades to a note.

## Retrieval tuning (2026-06-16)

A deterministic, LLM-free scorecard (`vida-eval/retrieval.py`) imports the server and asserts
RANKING expectations directly (right doc retrievable / near top / above a known-wrong one /
junk absent) — fast enough to run after every retrieval change, unlike the full LLM `run.py`.
Against it:

- **Weighted RRF fusion** (`_augment_with_vectors`). The old invariant "never demote a strong
  exact match" is kept by a regime switch on whether the lexical search found ANY strict
  (all-terms) match. Strict → additive (lexical order preserved, vectors fill empty slots).
  No strict match anywhere (loose relaxed-OR only) → weighted RRF lets a CONFIDENT vector hit
  outrank loose lexical junk ("Key warning" > "Ski holder" for "key won't turn"). All weights
  are env-tunable (`VIDA_RRF_*`); `VIDA_DISABLE_VECTORS=1` forces pure-FTS.
- **Lexical mis-ranks** (`_rerank`). Fetch a wider candidate set, then re-rank with two domain
  priors: reward query terms appearing in the TITLE, and on torque-intent queries boost the
  dedicated "Tightening torque(s)" spec tables (they carry the verified values). Plus a
  **bolt↔screw** synonym (VIDA's British "screw") that promotes the authoritative this-car
  spec table to a strict match.
- **Embedding precision gate** (item 3). A vector-only doc is injected only when its distance
  is below an absolute ceiling AND within a relative margin of the best vector hit, then must
  share a query term/synonym (or be very close) — a relative-to-best gate where a flat
  threshold couldn't separate signal from noise. Env-tunable (`VIDA_VEC_*`).
- **Cross-vehicle conflict flag** (`_torque_conflict`). On a torque-intent query, if shown
  this-car and other-scope docs disagree on a component's Nm value, the result carries a
  `conflict_warning` steering to this car's value (the M6 intake 19-vs-20 Nm case).

## What ships vs what stays local (the audit line)

| Ships (public repo) | Stays local (never committed) |
|---------------------|-------------------------------|
| `src/` ingestion + server code, `tests/`, `docs/`, `car.example.toml`, setup scripts | `data/vida-kb.sqlite3`, `mssql-data/`, `vida/`, extracted images/text, **embeddings/vectors**, `.env`, `car.toml` |

Before any publish: `git ls-files` audit for VIDA-derived content; test fixtures must stay
minimal/factual, never verbatim VIDA text.
