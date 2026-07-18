"""Embed the document corpus into a sidecar vector store for HYBRID retrieval.

Computes a dense embedding per document with a LOCAL, offline embedding model
(nomic-embed-text via an OpenAI-compatible endpoint — LM Studio or Ollama) and
stores the vectors in data/vida-vectors.sqlite3 using sqlite-vec. The MCP server
fuses these with FTS5/BM25 (reciprocal-rank fusion) so semantically-phrased queries
("make the AWD quieter" -> angle-gear docs) hit, without enumerating synonyms.

Vectors are VIDA-derived -> LOCAL ONLY, never shipped (same legal line as the store).
Idempotent: documents already embedded (by content hash) are skipped, so a re-run
resumes and only re-embeds changed docs.

Run (with an embedding endpoint up):
  uv run python src/build_embeddings.py
  EMBED_URL=http://localhost:11434/v1 EMBED_MODEL=nomic-embed-text uv run python src/build_embeddings.py
"""
import hashlib
import os
import sqlite3
import struct
import sys
import time
from pathlib import Path

import httpx
import sqlite_vec

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "vida-kb.sqlite3"
VEC_DB = ROOT / "data" / "vida-vectors.sqlite3"

EMBED_URL = os.environ.get("EMBED_URL", "http://localhost:1234/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
DIM = int(os.environ.get("EMBED_DIM", "768"))
BATCH = int(os.environ.get("EMBED_BATCH", "64"))
DOC_CHARS = 6000  # nomic context ~2048 tok; title + lead covers the discriminating text


def _embed(texts: list[str]) -> list[list[float]]:
    # nomic wants a task prefix; "search_document:" for the indexed side.
    payload = {"model": EMBED_MODEL, "input": [f"search_document: {t}" for t in texts]}
    r = httpx.post(f"{EMBED_URL}/embeddings", json=payload, timeout=120)
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


def _doc_text(title: str | None, text_md: str | None) -> str:
    return f"{(title or '').strip()}\n{(text_md or '')[:DOC_CHARS]}".strip()


def main():
    t0 = time.time()
    lite = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = lite.execute(
        "SELECT doc_id, title, substr(text_md,1,?) FROM documents", (DOC_CHARS,)
    ).fetchall()
    lite.close()

    vec = sqlite3.connect(VEC_DB)
    vec.execute("PRAGMA journal_mode=WAL")  # readers (the server) never block on the writer
    vec.enable_load_extension(True)
    sqlite_vec.load(vec)
    vec.enable_load_extension(False)
    vec.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_docs USING vec0(embedding float[{DIM}])")
    vec.execute(
        "CREATE TABLE IF NOT EXISTS vec_map ("
        " rowid INTEGER PRIMARY KEY, doc_id TEXT UNIQUE, hash TEXT)"
    )
    vec.commit()
    seen = {r[0]: r[1] for r in vec.execute("SELECT doc_id, hash FROM vec_map")}
    next_rowid = (vec.execute("SELECT COALESCE(MAX(rowid),0) FROM vec_map").fetchone()[0]) + 1

    todo = []
    for doc_id, title, text_md in rows:
        txt = _doc_text(title, text_md)
        h = hashlib.sha1(txt.encode("utf-8", "ignore")).hexdigest()[:16]
        if seen.get(doc_id) != h:
            todo.append((doc_id, txt, h))
    print(f"docs: {len(rows)} | already current: {len(rows)-len(todo)} | to embed: {len(todo)}",
          flush=True)

    done = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        try:
            embs = _embed([t for _, t, _ in chunk])
        except Exception as e:
            print(f"  embed batch at {i} failed ({e}); retrying once after 3s", flush=True)
            time.sleep(3)
            embs = _embed([t for _, t, _ in chunk])
        for (doc_id, _txt, h), emb in zip(chunk, embs):
            rid = seen_rowid = vec.execute(
                "SELECT rowid FROM vec_map WHERE doc_id=?", (doc_id,)).fetchone()
            if rid is None:
                rid = next_rowid
                next_rowid += 1
            else:
                rid = rid[0]
                vec.execute("DELETE FROM vec_docs WHERE rowid=?", (rid,))
            blob = struct.pack(f"{DIM}f", *emb)
            vec.execute("INSERT INTO vec_docs(rowid, embedding) VALUES (?,?)", (rid, blob))
            vec.execute("INSERT OR REPLACE INTO vec_map(rowid, doc_id, hash) VALUES (?,?,?)",
                        (rid, doc_id, h))
        done += len(chunk)
        if i % (BATCH * 20) == 0:
            vec.commit()
            print(f"  {done}/{len(todo)}  ({done/(time.time()-t0):.0f}/s)", flush=True)
    vec.commit()
    total = vec.execute("SELECT COUNT(*) FROM vec_map").fetchone()[0]
    vec.close()
    print(f"done in {time.time()-t0:.0f}s | {total} vectors in {VEC_DB.name}", flush=True)


if __name__ == "__main__":
    main()
