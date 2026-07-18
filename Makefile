# vida-kb — one-command ingestion + ops (item 4)
#
# The repo ships CODE; the DATA is built locally from YOUR licensed VIDA install
# (nothing VIDA-derived is committed). `make ingest` runs the whole pipeline in order.
# Each step is also a target so you can re-run just one. Re-runs are incremental where
# the script supports it (extract_docs hashes; build_embeddings hashes).
#
#   make check      # preflight: is vida-sql up? embed endpoint up? which artifacts exist?
#   make ingest     # full build: store -> docs -> EPC -> synonyms -> images -> embeddings
#   make refresh    # cheap incremental refresh of docs (hash-skip unchanged)
#   make serve      # run the MCP server (+ figure image server) on :8765
#   make eval       # retrieval scorecard + engine-scope + incremental tests

PY := uv run python
.PHONY: ingest refresh store docs epc synonyms images embeddings serve eval test check clean-artifacts help dataset
.DEFAULT_GOAL := help

## ── full pipeline ────────────────────────────────────────────────────────────
# Order matters: build_db creates the store + FTS; extract_docs fills documents;
# extract_epc adds EPC parts + generic_docs and REBUILDS FTS; the rest are independent
# sidecars. Images need the imagerepository DB attached; embeddings need an embed endpoint.
ingest: store docs epc synonyms images embeddings
	@echo "ingest complete — run 'make check' to confirm all artifacts, then 'make serve'."

store:
	$(PY) src/build_db.py

docs:
	$(PY) src/extract_docs.py $(ARGS)   # incremental by default; 'make docs ARGS=--full' to rebuild

refresh:
	$(PY) src/extract_docs.py        # hash-skips unchanged docs (cheap refresh)

epc:
	$(PY) src/extract_epc.py

synonyms:
	$(PY) src/build_synonyms.py

images:
	$(PY) src/extract_images.py

embeddings:
	$(PY) src/build_embeddings.py

# Optional (CGM Milestone 2): render the few document-referenced vector diagrams to SVG.
# Needs a converter (VIDA_CGM_CMD, e.g. jcgm — see docs/images-plan.md); no-op without one.
# NOT in `ingest` because almost all document figures are raster (extract_images covers them).
cgm:
	$(PY) src/extract_cgm.py

## ── run / verify ─────────────────────────────────────────────────────────────
serve:
	$(PY) src/server.py --http 8765

eval test:
	$(PY) vida-eval/retrieval.py
	$(PY) tests/test_engine_scope.py
	$(PY) tests/test_incremental.py
	$(PY) tests/test_dataset_realism.py

## ── dataset (MAC-SIDE only) ──────────────────────────────────────────────────
# Build the function-calling training JSONL with REAL multi-hit tool results (the
# live retriever -> train == serve). Needs the corpus + embeddings, so it runs HERE,
# never on the trainer box; ship the result with scripts/cloud/push_to_box.sh.
#   make dataset                  # default size
#   make dataset DATASET_LIMIT=4000
DATASET_LIMIT ?= 2000
dataset:
	$(PY) -m dataset.gen_dataset --limit $(DATASET_LIMIT) --out data/ft/train.jsonl


# Preflight: SQL Server reachable? embed endpoint reachable? which artifacts are built?
# Reads the same status block list_sources() exposes, so it matches runtime reality.
check:
	@$(PY) -c "import sys; sys.path.insert(0,'src'); import server, json; \
print(json.dumps(server.list_sources()['artifacts'], indent=2))"

clean-artifacts:
	@echo "removing locally-built (gitignored) artifacts — VIDA data is re-buildable via 'make ingest'"
	rm -f data/vida-kb.sqlite3 data/vida-images.sqlite3 data/vida-vectors.sqlite3 \
	      data/*.sqlite3-wal data/*.sqlite3-shm

help:
	@grep -E '^(##|[a-z-]+:)' Makefile | sed -e 's/^## //' -e 's/:.*//' | sed 's/^/  /'
	@echo "  (see 'make check' for artifact status; full setup walkthrough in README.md)"
