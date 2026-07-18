# VIDAR one-shot setup — hand this to Claude Code (automode) and walk away

You are setting up a complete, local, grounded Volvo **VIDA** service assistant on THIS machine,
end to end, with **zero manual work from the user**. The only precondition is that the user's
licensed VIDA install is present on the machine — you locate it, read what you need from it, and do
everything else yourself: install tools, download the model, build the index, wire Open WebUI, verify.

**Hard rules:**
- Ship code, not data. NEVER commit, upload, or transmit any VIDA content or the user's credentials.
- Read the VIDA SQL credentials from the install's own config at runtime (step 1); keep them only in
  the gitignored `.env`, never echo, log, or hardcode them anywhere in this repo.
- The assistant identifies vehicles by **model + year + engine code**, never a VIN (no VIN decoder exists).
- SQL access is read-only + allowlisted. Never enable model-authored SQL.

## Do this, in order — report progress at each step

### 0. Discover & install prerequisites (no user action)
- Ensure **Docker**, **Ollama**, and **uv** are installed; install any that are missing (Homebrew on macOS).
- Ensure an **Open WebUI** instance (install via `uv tool install open-webui` or reuse a running one; default `:3000`).
- **Locate the VIDA install automatically**: search common paths for the VIDA `*.mdf` database files
  (EPC, servicerep_en-US, CarCom, BaseData, DiagSwdlRepository, imagerepository). Only if you can't
  find them, ask the user for the path. Confirm which DBs are present.

### 1. Read the SQL credentials from the install (no user input)
VIDA hardcodes its SQL Server credentials in the install's own config. Read them directly — do NOT
ask the user and do NOT hardcode any value in this repo:
- Parse `VidaConfigApplication.exe.Config` (and the `install/vida.config.datasources.xml` /
  `System/config/` set) in the located install for the SQL connection string / credentials element.
- Extract the user + password from it in memory only. Use them ONLY to attach DBs and create the
  read-only login. Write them to the gitignored `.env` and NOWHERE else. Never print or log the password.

### 2. VIDA SQL Server (automated)
- Run a SQL Server 2022 container `vida-sql` (restart policy `no`).
- Copy the located `*.mdf` files into `mssql-data/` (working copies; leave originals in the VIDA install).
- Attach them (`src/attach_dbs.sql`).
- Create a read-only login and verify writes are blocked:
  `CREATE LOGIN vida_ro WITH PASSWORD='<generate>', CHECK_POLICY=OFF;` + `db_datareader` in each DB.
- Write `.env` from `.env.example`: `VIDA_SQL_USER=vida_ro`, the generated password, server/port.

### 3. Model + embeddings (automated, from Hugging Face)
- Pull the model: `ollama pull hf.co/slingj/VIDAR-Grounded-7B:Q4_K_M` (or `ollama create` from the
  downloaded GGUF) with `temperature 0.1`, `num_ctx 8192`, GGUF-inferred template.
- Pull the embedding model: `ollama pull nomic-embed-text`. Set `EMBED_URL=http://localhost:11434/v1`,
  `EMBED_MODEL=nomic-embed-text` in `.env`.

### 4. Build the knowledge base (automated)
- `make check` (preflight: SQL up? embed endpoint up?), then `make ingest`
  (`build_db → extract_docs → extract_epc → build_synonyms → extract_images → build_embeddings`).
- Report per-step row counts. Re-runs are incremental (docs + embeddings hash-skip unchanged).

### 5. Serve + wire Open WebUI (automated)
- `make serve` (MCP server on `:8765`, streamable-HTTP) in the background.
- In Open WebUI (via its API/config): register the MCP endpoint (native MCP or `mcpo`), set the grounded
  system prompt from `docs/system-prompt.md`, install the VCC→link citation outlet filter from
  `docs/openwebui/`, and select the `VIDAR-Grounded-7B` model.

### 6. Verify, then hand back
- Send a known vehicle-specific question with a `[VEHICLE] <year> <model> | <engine> [/VEHICLE]` block;
  confirm the model calls a tool and answers with a `(VCC-…)` citation.
- Run `make eval` (retrieval scorecard). Report the Open WebUI URL and a one-line "ready" summary.
- If `vida-sql` is later down, normal chat still works off the baked SQLite stores; it's only needed to re-ingest.

**Success = the user opens Open WebUI, picks VIDAR-Grounded-7B, asks about their car, and gets a cited answer — having done nothing but have their VIDA install present on the machine.**
