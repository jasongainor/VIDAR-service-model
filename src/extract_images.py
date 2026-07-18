"""Bake the referenced VIDA figures into a self-contained local blob store.

Pulls every raster image referenced by an indexed document (documents.image_refs)
out of the live `imagerepository` SQL Server and into a sidecar SQLite DB
(data/vida-images.sqlite3). After this runs the MCP server serves figures from
that file, so the heavyweight SQL Server is a SETUP-time dependency only — the
runtime needs nothing but the artifact. Idempotent: already-stored paths are
skipped, so a re-run resumes where it left off.

Run (with vida-sql up and .env credentials present):
  uv run python src/extract_images.py
"""
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "vida-kb.sqlite3"
IMAGES_DB = ROOT / "data" / "vida-images.sqlite3"

_CTYPE = {".gif": "image/gif", ".jpeg": "image/jpeg", ".jpg": "image/jpeg", ".png": "image/png"}


def _env(key: str) -> str | None:
    if key in os.environ:
        return os.environ[key]
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return None


def _referenced_rasters() -> list[str]:
    lite = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    refs: set[str] = set()
    for (ir,) in lite.execute(
        "SELECT image_refs FROM documents WHERE image_refs IS NOT NULL"
        " AND image_refs != '[]' AND image_refs != ''"
    ):
        for p in json.loads(ir or "[]"):
            if os.path.splitext(p)[1].lower() in _CTYPE:
                refs.add(p)
    lite.close()
    return sorted(refs)


def main():
    t0 = time.time()
    import pytds

    user, pw = _env("VIDA_SQL_USER"), _env("VIDA_SQL_PASSWORD")
    if not user or not pw:
        sys.exit("VIDA_SQL_USER / VIDA_SQL_PASSWORD missing (.env)")

    out = sqlite3.connect(IMAGES_DB)
    out.execute("PRAGMA journal_mode=WAL")  # readers (the server) never block on the writer
    out.execute(
        "CREATE TABLE IF NOT EXISTS images ("
        " path TEXT PRIMARY KEY, content_type TEXT, data BLOB)"
    )
    out.commit()
    have = {r[0] for r in out.execute("SELECT path FROM images")}

    rasters = _referenced_rasters()
    todo = [p for p in rasters if p not in have]
    print(f"referenced rasters: {len(rasters)} | already stored: {len(have)} | to fetch: {len(todo)}",
          flush=True)

    con = pytds.connect(server="127.0.0.1", port=1433, database="imagerepository",
                        user=user, password=pw, autocommit=True, login_timeout=5, timeout=30)
    cur = con.cursor()
    fetched = missing = 0
    for i, path in enumerate(todo, 1):
        cur.execute("SELECT imageData FROM LocalizedGraphics WHERE path = %s", (path,))
        row = cur.fetchone()
        if row and row[0]:
            ctype = _CTYPE[os.path.splitext(path)[1].lower()]
            out.execute("INSERT OR REPLACE INTO images VALUES (?,?,?)",
                        (path, ctype, bytes(row[0])))
            fetched += 1
        else:
            missing += 1
        if i % 2000 == 0:
            out.commit()
            print(f"  {i}/{len(todo)}  fetched={fetched} missing={missing}  "
                  f"({i/(time.time()-t0):.0f}/s)", flush=True)
    out.commit()

    # Bake the figure marker title -> path map too, so get_figures resolves non-1:1
    # docs locally and needs NO live SQL at all. Metadata-only (fast).
    out.execute("CREATE TABLE IF NOT EXISTS graphics (title TEXT, path TEXT, PRIMARY KEY (title, path))")
    out.execute("DELETE FROM graphics")
    have_paths = [r[0] for r in out.execute("SELECT path FROM images")]
    gmap = 0
    for i in range(0, len(have_paths), 1000):
        chunk = have_paths[i:i + 1000]
        ph = ",".join(["%s"] * len(chunk))
        cur.execute(f"SELECT title, path FROM LocalizedGraphics WHERE path IN ({ph})", tuple(chunk))
        rows = [(t, p) for t, p in cur.fetchall() if t]
        out.executemany("INSERT OR IGNORE INTO graphics VALUES (?,?)", rows)
        gmap += len(rows)
    out.execute("CREATE INDEX IF NOT EXISTS idx_graphics_title ON graphics(title)")
    out.commit()
    con.close()
    print(f"graphics title->path map: {gmap} rows", flush=True)

    total = out.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    size_mb = (out.execute("SELECT SUM(LENGTH(data)) FROM images").fetchone()[0] or 0) / 1e6
    out.close()
    print(f"done in {time.time()-t0:.0f}s | stored {total} images, ~{size_mb:.0f} MB | "
          f"this run fetched {fetched}, {missing} not in repo", flush=True)


if __name__ == "__main__":
    main()
