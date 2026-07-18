# Open WebUI — explicit entry for the trained VIDA model

The fine-tuned champion (`vida-qwen2.5-7b-r16`, the 85.3%@75 grounded model) is installed in
ollama as a self-contained model named **`vidar-grounded-7b`** — the grounding system prompt
(train==serve contract) is baked in, temperature 0, num_ctx 32768. Re-run anytime with:

    .venv/bin/python scripts/cloud/install_owui_model.py        # -> ollama model 'vidar-grounded-7b'

It is self-contained: it calls tools / grounds / refuses correctly even with no client-side
system prompt (verified). Pick a different config with `install_owui_model.py <name> <tag>`
(e.g. `vidar-grounded-7b-r32 qwen25-7b-r32`).

## Make it a selectable model in Open WebUI

1. **Connect OWUI to this ollama.** Admin → Settings → Connections → Ollama API →
   `http://<this-mac-private-ip>:11434`. `vidar-grounded-7b` then shows in the model list.
2. **Create the model entry.** Workspace → Models → **+ New Model**:
   - **Base model:** `vidar-grounded-7b`
   - **Name:** `VIDAR Grounded 7B`
   - **System prompt:** leave **empty** (baked into the model) — or paste
     `grounding/contract.py:system_msg()` if you want it visible/editable in OWUI.
   - **Tools:** attach the **VIDA MCP tools** (`search_docs`, `get_document`, `epc_part`,
     `lookup_pin`, `get_figures`). Without them the model is grounded-but-blind **by design**
     (it knows zero facts; retrieval supplies every fact).
   - **Filter:** enable **VIDA Citation Links** (`vida_citation_links.py`) for clickable sources.
   - **Params:** temperature `0`, num_ctx `32768` (already baked; set here too if you like).
3. **Save.** `VIDAR Grounded 7B` is now selectable in the chat model dropdown.

## Notes

- **Which ollama?** The model is installed in the ollama on the machine where you ran the
  installer. If your OWUI talks to a different ollama host, run the installer there (it only
  needs the GGUF under `cloud-artifacts/.../blobs/` + this repo), or point OWUI at this host.
- **Tools are mandatory.** The whole point is grounding — a `vidar-grounded-7b` entry with no MCP
  tools attached will refuse most car questions (it has no facts to ground on). Reuse the same
  MCP tool connection your existing VIDA RAG model uses.
- **Reachability** of the `/doc` citation links is private-network-only (same as figures).
