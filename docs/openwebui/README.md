# Open WebUI — clickable VIDA citations

The grounded model cites sources as bare VIDA tokens like `VCC-461742-2`. By default Open
WebUI renders those as **plain text**, so there's no way to click through and verify the
source document. These two pieces make citations clickable:

## 1. Server side — the `/doc/<ref>` route (already in `src/server.py`)

The MCP server's private-network-bound HTTP server (the same one that serves figures at `/img/`)
now also serves `/doc/<VCC-ref>`, rendering the cited document's text as a readable HTML
page. It is read-only, validates the ref against a strict pattern, and HTML-escapes the
body (no injection surface). It comes up automatically when the figure server starts
(`_start_image_server`, host/port = `VIDA_IMAGE_HOST` / `VIDA_IMAGE_PORT`).

Verify it's live (from a client on the same private network):

    curl -s http://<VIDA_IMAGE_HOST>:<VIDA_IMAGE_PORT>/doc/VCC-453588-1 | head

Tool results (`get_document`) also now carry a `ref_url` field pointing at this route.

## 2. Open WebUI side — the citation-link outlet filter

`vida_citation_links.py` is an Open WebUI **Filter** function. Its `outlet` rewrites every
`VCC-######(-#)` token in the answer body into a Markdown link to `<doc_base_url>/doc/<ref>`.
It is idempotent (won't double-link) and changes nothing else.

**Install:** Open WebUI → Admin Panel → Functions → **+ New Function** → paste
`vida_citation_links.py` → Save → enable (globally, or per-model on the VIDA model). Then set
the Valve **`doc_base_url`** to your VIDA doc server base (default is the private-network
`host:port` the figure server already uses).

## Notes / decisions

- **Reachability:** links resolve only where the doc server is reachable — i.e. over the
  private network (the iPhone PWA, other devices on it), exactly like the existing figure
  `/img/` links. Widening the bind beyond the private interface is a deployment choice, not done here.
- **No retraining:** this deliberately does NOT change `grounding/contract.py:citation()`.
  Emitting Markdown links from the model would alter the shared train==serve contract
  (`CONTRACT_VERSION`) and require regenerating the dataset + retraining. The outlet achieves
  clickable sources without touching the model.
