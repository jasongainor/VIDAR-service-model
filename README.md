# VIDAR — service model

A grounded Volvo **VIDA** service assistant you run on your own hardware: the [`slingj/VIDAR-Grounded-7B`](https://huggingface.co/slingj/VIDAR-Grounded-7B) model + the retrieval harness that turns **your own licensed VIDA install** into a local, citable knowledge base for Open WebUI.

> **Ships code, not data.** Nothing VIDA-derived is committed or redistributed. You point this at *your* licensed VIDA installation and it builds a local search index on *your* machine. The credentials that unlock the VIDA databases are **unique to your install** — you supply your own; none ship here.

This repo is a shell + prompts. Hand it (and the bootstrap prompt) to a coding agent and it walks the whole setup with no manual SQL.

## What you provide
Just a **licensed VIDA install** present on the machine — its SQL Server database files (`*.mdf`). The
agent reads the install's own SQL credentials, and installs everything else it needs: Ollama (chat +
`nomic-embed-text` embeddings), Open WebUI, and the model. (Running it by hand instead? You'll want those
three in place — see "Or run it yourself" below.)

## The pipeline (what actually happens)
Verified against the reference implementation:

1. **Stand up the VIDA SQL Server** — run a SQL Server 2022 container (`vida-sql`), attach working copies of your VIDA `*.mdf` databases (EPC, servicerep, CarCom, BaseData, DiagSwdlRepository, imagerepository) from `mssql-data/` via `src/attach_dbs.sql`. Originals stay in your VIDA install.
2. **Set credentials** — copy `.env.example` → `.env`, set your VIDA SQL Server user/password (gitignored, never committed). Setup uses `sa`; runtime tools use a read-only `vida_ro` login (write-blocked).
3. **Ingest** (`make ingest`, order matters):
   `build_db` (SQLite store + FTS5) → `extract_docs` (service docs) → `extract_epc` (parts catalog + lookup tables) → `build_synonyms` (VIDA Lexicon) → `extract_images` (bake referenced figures) → `build_embeddings` (nomic-embed → `vida-vectors.sqlite3`).
4. **Serve** (`make serve`) — the MCP server on `:8765` (streamable-HTTP for Open WebUI / `mcpo`), exposing the grounded tools: `search_docs`, `get_document`, `epc_part`, `lookup_pin`, `get_figures`, `list_sources`.
5. **Wire Open WebUI** — register the MCP endpoint, set the grounded system prompt, add the VCC→link citation outlet filter (`docs/openwebui/`), and select `VIDAR-Grounded-7B`.

After ingest the runtime is **self-contained** — the SQLite stores are baked, so `vida-sql` is only needed for setup/refresh (its restart policy is `no`; `docker start vida-sql` when you re-ingest).

## One command — hand it to Claude Code and walk away
The intended path. Clone this repo, open it in **Claude Code (automode)**, and give it the prompt:

```sh
git clone https://github.com/jasongainor/VIDAR-service-model && cd VIDAR-service-model
claude   # then: "follow prompts/bootstrap.md"
```

From there the agent does **everything** — installs Docker/Ollama/Open WebUI if missing, finds your
VIDA install, stands up the SQL container and attaches the DBs, downloads `VIDAR-Grounded-7B` from
Hugging Face, builds the local index, serves the MCP endpoint, wires it into Open WebUI, and verifies
a cited answer. See [`prompts/bootstrap.md`](prompts/bootstrap.md).

**The only thing you supply**: your **licensed VIDA install** present on the machine. The agent locates
it, reads the SQL credentials from the install's own config (`VidaConfigApplication.exe.Config`), builds
the index, and wires everything up — you don't hand it a password or run a single command yourself.
Nothing VIDA-derived and no credential ever leaves your machine.

## Or run it yourself
```sh
cp .env.example .env          # set YOUR VIDA SQL credentials
docker start vida-sql         # one-time container setup (bootstrap.md does this for you)
make check                    # preflight: SQL up? embed endpoint up? artifacts?
make ingest                   # build the local knowledge base from your VIDA
make serve                    # MCP server on :8765
```
Then point Open WebUI at `http://localhost:8765` and run `VIDAR-Grounded-7B` in Ollama.

## Notes verified from the build
- **No VIN decode.** VIDA's VIN→profile tables were never located during schema discovery; the assistant identifies vehicles by **model + year + engine code**, not VIN — by design.
- **Read-only + allowlisted.** SQL tools use fixed parameterized templates against a read-only login; the model never writes SQL (prompt-injection containment).

## How the model was made — and how to reproduce or challenge it

The repo carries the pieces that define the model's behaviour, not just the harness that serves it:

- **[`grounding/contract.py`](grounding/contract.py)** — the exact message shape the model is trained and served under. If you fine-tune or re-serve, match this or results won't. This is the one file to read first.
- **[`dataset/`](dataset/)** — the generator that turns *your* licensed VIDA install into the training JSONL, plus [`DATASET_CARD.md`](dataset/DATASET_CARD.md). No rows are distributed; you produce your own.
- **[`vida-eval/`](vida-eval/)** — the frozen benchmark and its graders. Part numbers, torque values and refusals are graded **deterministically** (a wrong torque is a safety failure, not a style nit); only procedure-correctness escalates to a model judge.
- **[`TRAINING.md`](TRAINING.md)** — the QLoRA recipe: hyperparameters, the per-family response-masking markers, and the sequence-length finding. The trainer itself isn't shipped — it was thin glue over fast-moving APIs, and a stale trainer is worse than an accurate recipe.
- **[`publish/`](publish/)** — model + card upload to Hugging Face.

The eval matters more than it looks: the numbers on the model card are unverifiable without it, and it's what tells you whether a quantized artifact still holds up.

## License
See `LICENSE` — released in the spirit of the open Volvo interop/research tooling community.
