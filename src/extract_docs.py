"""Extract VIDA Information documents from servicerep_en-US into the SQLite store.

Pipeline (see docs/schema-map.md):
  Document.XmlContent (PKZIP of one XML) -> XML -> markdown + image refs
  DocumentProfile + BaseData lookups     -> profile/model/year/engine tags
  TreeItem breadcrumbs + QualifierGroup  -> section metadata

Usage:
  uv run python src/extract_docs.py --sample 50   # trial run, prints a preview
  uv run python src/extract_docs.py               # full extraction (~86.5k docs)
"""
import argparse
import hashlib
import io
import json
import re
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

import pytds
from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "vida-kb.sqlite3"
SRC_DB = "servicerep_en-US"
from dbconfig import conn_params  # SQL creds from .env (see .env.example) — never hardcoded
CONN = conn_params()

EXTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
  profile_id TEXT PRIMARY KEY,
  description TEXT, model TEXT, year TEXT, engine TEXT, folder_level INTEGER
);
CREATE TABLE IF NOT EXISTS doc_profiles (
  doc_id TEXT, profile_id TEXT,
  PRIMARY KEY (doc_id, profile_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_doc_profiles_profile ON doc_profiles(profile_id);
-- Incremental-rebuild ledger (item 5): the content hash of each doc's SOURCE blob, so a
-- re-run can skip docs whose XmlContent is byte-identical and only re-parse what changed.
CREATE TABLE IF NOT EXISTS doc_hashes (
  doc_id TEXT PRIMARY KEY, hash TEXT
) WITHOUT ROWID;
"""


def _doc_hash(blob: bytes) -> str:
    """Stable short content hash of a document's source blob (the PKZIP'd XmlContent).
    Hashing the SOURCE bytes lets us decide to skip BEFORE the expensive unzip+XML parse."""
    return hashlib.sha1(bytes(blob)).hexdigest()[:16]


def plan_prune(seen_ids: set[str], current_ids: set[str], full: bool) -> set[str]:
    """doc_ids to delete on an incremental run: rows in the store whose doc no longer
    exists in the source. On a --full rebuild nothing is pruned here (the table is wiped
    up front instead). Pure + side-effect-free so it can be unit-tested without SQL Server."""
    return set() if full else (seen_ids - current_ids)

# ---------------------------------------------------------------- XML -> md

INLINE_TAGS = {"emph", "ptxt2", "sub", "sup", "ref", "xref", "href"}
SKIP_TAGS = {"meta", "metadata"}


def _inline_text(el) -> str:
    """Flatten an element to inline markdown text."""
    parts = []
    if el.text:
        parts.append(el.text)
    for child in el:
        tag = etree.QName(child).localname if isinstance(child.tag, str) else ""
        if tag == "graphic":
            pass  # handled at block level; refs collected separately
        elif tag == "href":
            title = child.get("title")
            inner = _inline_text(child).strip()
            if title:
                parts.append(f"[{title}]")
            elif not re.match(r"(en-[A-Za-z]{2}|0[89a-f]00c8af)", inner):
                parts.append(f"[{inner}]")  # suppress raw chronicle-id link targets
        elif tag == "emph":
            parts.append(f"**{_inline_text(child).strip()}** ")
        elif tag in ("ptxt", "para", "p"):
            parts.append(" " + _inline_text(child).strip() + " ")
        else:
            parts.append(_inline_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _render(el, depth, out, images):
    tag = etree.QName(el).localname if isinstance(el.tag, str) else ""
    if tag in SKIP_TAGS:
        return
    if tag == "graphic":
        for href in el.iter():
            t = etree.QName(href).localname if isinstance(href.tag, str) else ""
            if t == "href" and href.text:
                images.append(href.text.strip())
                title = href.get("title")
                if title:
                    out.append(f"_[image {title}]_")
        return
    if tag == "title":
        text = _inline_text(el).strip()
        if text:
            out.append(f"{'#' * min(depth + 1, 6)} {text}")
        return
    if tag in ("ptxt", "para", "p"):
        text = _inline_text(el).strip()
        if text:
            out.append(text)
        return
    if tag == "item":
        text = _inline_text(el).strip()
        sub_blocks = [c for c in el if isinstance(c.tag, str)
                      and etree.QName(c).localname not in INLINE_TAGS | {"graphic"}]
        if text:
            out.append(f"- {text}")
        for c in sub_blocks:
            t = etree.QName(c).localname
            if t in ("list1", "list2", "ptxt"):
                continue  # already flattened into text above
            _render(c, depth, out, images)
        return
    if tag in ("warning", "caution", "note", "hint"):
        text = _inline_text(el).strip()
        if text:
            out.append(f"**{tag.upper()}:** {text}")
        return
    if tag in ("row",):
        cells = []
        for c in el:
            if isinstance(c.tag, str) and etree.QName(c).localname == "entry":
                cells.append(_inline_text(c).strip() or " ")
        if cells:
            out.append("| " + " | ".join(cells) + " |")
        return
    if tag in ("xref", "ref", "href"):
        text = _inline_text(el).strip()
        title = el.get("title") or next((h.get("title") for h in el.iter() if h.get("title")), None)
        if title:
            out.append(f"[{title}]")
        elif text and not re.match(r"(en-[A-Za-z]{2}|0[89a-f]00c8af)", text):
            out.append(f"[{text}]")
        return
    if tag in ("component", "value", "tightening"):  # torqueinfo leaves
        unit = el.get("unit", "")
        text = _inline_text(el).strip()
        if text:
            out.append(f"{tag}: {text} {unit}".rstrip())
        return
    # structural containers: recurse with deeper headings where sensible
    deeper = tag in ("procedure", "stepgrp", "procstep", "section", "subsection", "spec", "specgrp")
    if el.text and el.text.strip() and tag not in ("servinfosub", "diagnostic"):
        out.append(el.text.strip())
    for child in el:
        if isinstance(child.tag, str):
            _render(child, depth + (1 if deeper else 0), out, images)
        if child.tail and child.tail.strip():
            out.append(child.tail.strip())


def xml_to_markdown(xml_bytes: bytes) -> tuple[str, list[str], dict]:
    root = etree.fromstring(xml_bytes)
    out: list[str] = []
    images: list[str] = []
    _render(root, 0, out, images)
    images = list(dict.fromkeys(images))
    md = "\n\n".join(s for s in (re.sub(r"[ \t]+", " ", b).strip() for b in out) if s)
    attrs = {"docno": root.get("docno"), "ie_id": root.get("IE-ID"), "root": etree.QName(root).localname}
    return md, images, attrs


def unzip_xml(blob: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        return z.read(name)


# ---------------------------------------------------------------- loaders

def load_profiles(con_lite):
    """BaseData.VehicleProfile -> profiles table."""
    sql = """
    SELECT vp.Id,
           ltrim(rtrim(coalesce(vm.Description,''))),
           ltrim(rtrim(coalesce(my.Description,''))),
           ltrim(rtrim(coalesce(e.Description,''))),
           vp.FolderLevel
    FROM dbo.VehicleProfile vp
    LEFT JOIN dbo.VehicleModel vm ON vm.Id = vp.fkVehicleModel
    LEFT JOIN dbo.ModelYear   my ON my.Id = vp.fkModelYear
    LEFT JOIN dbo.Engine      e  ON e.Id  = vp.fkEngine
    """
    with pytds.connect(database="BaseData", **CONN) as c:
        cur = c.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
    con_lite.execute("DELETE FROM profiles")
    for pid, model, year, engine, lvl in rows:
        desc = ", ".join(x for x in (model, year, engine) if x)
        con_lite.execute(
            "INSERT OR REPLACE INTO profiles VALUES (?,?,?,?,?,?)",
            (pid.strip(), desc, model or None, year or None, engine or None, lvl),
        )
    n = con_lite.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    print(f"profiles: {n}", flush=True)


def load_doc_profiles(con_lite):
    con_lite.execute("DELETE FROM doc_profiles")
    with pytds.connect(database=SRC_DB, **CONN) as c:
        cur = c.cursor()
        cur.execute("SELECT fkDocument, profileId FROM dbo.DocumentProfile")
        while True:
            batch = cur.fetchmany(50000)
            if not batch:
                break
            con_lite.executemany(
                "INSERT OR IGNORE INTO doc_profiles VALUES (?,?)",
                ((f"{SRC_DB}:Document:{fk}", pid.strip()) for fk, pid in batch),
            )
    n = con_lite.execute("SELECT COUNT(*) FROM doc_profiles").fetchone()[0]
    print(f"doc_profiles: {n}", flush=True)


def load_lookups():
    with pytds.connect(database=SRC_DB, **CONN) as c:
        cur = c.cursor()
        cur.execute("SELECT id, name FROM dbo.DocumentType")
        doc_types = dict(cur.fetchall())
        cur.execute(
            "SELECT q.id, qg.name FROM dbo.Qualifier q JOIN dbo.QualifierGroup qg ON qg.id = q.fkQualifierGroup"
        )
        qual_group = dict(cur.fetchall())
        cur.execute(
            "SELECT tid.projectDocumentTo, t.title FROM dbo.TreeItemDocument tid JOIN dbo.TreeItem t ON t.id = tid.fkTreeItem WHERE t.title IS NOT NULL"
        )
        tree = {}
        for pd, title in cur.fetchall():
            pd = pd.strip()
            titles = tree.setdefault(pd, [])
            t = title.strip()
            if t and t not in titles and len(titles) < 4:
                titles.append(t)
    return doc_types, qual_group, tree


def extract(limit: int | None, preview: bool, full: bool = False):
    con_lite = sqlite3.connect(DB_PATH)
    con_lite.executescript(EXTRA_SCHEMA)
    doc_types, qual_group, tree = load_lookups()
    print(f"lookups: {len(doc_types)} types, {len(qual_group)} qualifiers, {len(tree)} tree-mapped docs", flush=True)

    seen_hashes: dict[str, str] = {}
    if not preview:
        load_profiles(con_lite)
        load_doc_profiles(con_lite)
        if full:
            con_lite.execute("DELETE FROM documents")
            con_lite.execute("DELETE FROM raw_documents")
            con_lite.execute("DELETE FROM doc_hashes")
        else:
            # incremental (item 5): keep existing rows; the hash ledger drives skip/refresh
            seen_hashes = dict(con_lite.execute("SELECT doc_id, hash FROM doc_hashes"))
            print(f"incremental: {len(seen_hashes)} docs already hashed (use --full to rebuild)", flush=True)
        con_lite.commit()

    top = f"TOP {limit} " if limit else ""
    sql = (
        f"SELECT {top}id, chronicleId, projectDocumentId, vccNumber, title,"
        " fkDocumentType, fkQualifier, XmlContent FROM dbo.Document"
    )
    n_ok = n_err = n_skip = 0
    seen_ids: set[str] = set()
    t0 = time.time()
    with pytds.connect(database=SRC_DB, **CONN) as c:
        cur = c.cursor()
        cur.execute(sql)
        while True:
            batch = cur.fetchmany(200)
            if not batch:
                break
            for (did, chron, pdoc, vcc, title, ftype, fqual, blob) in batch:
                doc_id = f"{SRC_DB}:Document:{did}"
                seen_ids.add(doc_id)
                h = _doc_hash(blob)
                if not preview and not full and seen_hashes.get(doc_id) == h:
                    n_skip += 1  # byte-identical source — skip the unzip+parse entirely
                    continue
                try:
                    xml_bytes = unzip_xml(bytes(blob))
                    md, images, attrs = xml_to_markdown(xml_bytes)
                except Exception as e:  # noqa: BLE001 — keep the row, degrade to raw text
                    n_err += 1
                    try:
                        xml_bytes = unzip_xml(bytes(blob))
                        md = re.sub(r"<[^>]+>", " ", xml_bytes.decode("utf-8", "replace"))
                        md = re.sub(r"\s+", " ", md).strip()
                        images, attrs = [], {}
                    except Exception:
                        print(f"FAILED {doc_id}: {e}", file=sys.stderr, flush=True)
                        continue
                pdoc_s = (pdoc or "").strip()
                breadcrumb = " | ".join(tree.get(pdoc_s, []))
                section = qual_group.get(fqual) or doc_types.get(ftype, "")
                if not (title or "").strip():
                    # torqueinfo etc. carry no title; derive one so citations stay readable
                    first = md.replace("component: ", "").replace("\n\nvalue: ", " — ").split("\n", 1)[0]
                    title = f"{doc_types.get(ftype, 'document')}: {first[:80]}" if first else vcc
                if breadcrumb:
                    md = f"_Section: {breadcrumb}_\n\n{md}"
                if preview and n_ok < 3:
                    print("=" * 70)
                    print(f"{doc_id}  [{vcc}]  type={doc_types.get(ftype)} section={section}")
                    print(md[:1200])
                    print(f"images: {images[:5]}")
                con_lite.execute(
                    "INSERT OR REPLACE INTO documents (doc_id, db, table_name, source_pk,"
                    " vida_doc_ref, title, section, text_md, image_refs)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (doc_id, SRC_DB, "Document", str(did), vcc, title, section, md,
                     json.dumps(images)),
                )
                if not preview:
                    con_lite.execute(
                        "INSERT OR REPLACE INTO raw_documents (doc_id, xml, image_refs) VALUES (?,?,?)",
                        (doc_id, xml_bytes.decode("utf-8", "replace"), json.dumps(images)),
                    )
                    con_lite.execute(
                        "INSERT OR REPLACE INTO doc_hashes (doc_id, hash) VALUES (?,?)",
                        (doc_id, h),
                    )
                n_ok += 1
            con_lite.commit()
            if n_ok % 5000 < 200:
                rate = n_ok / max(time.time() - t0, 1e-9)
                print(f"  {n_ok} docs ({rate:.0f}/s, {n_err} fallbacks, {n_skip} unchanged)", flush=True)

    # Prune rows whose source document no longer exists (incremental only).
    n_prune = 0
    if not preview:
        prune = plan_prune(set(seen_hashes), seen_ids, full)
        for doc_id in prune:
            con_lite.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            con_lite.execute("DELETE FROM raw_documents WHERE doc_id = ?", (doc_id,))
            con_lite.execute("DELETE FROM doc_hashes WHERE doc_id = ?", (doc_id,))
            n_prune += 1
        con_lite.commit()

    print(f"extracted {n_ok} docs ({n_skip} unchanged, {n_prune} pruned), "
          f"{n_err} normalize-fallbacks in {time.time()-t0:.0f}s", flush=True)

    if not preview and (n_ok or n_prune):
        # Nothing parsed and nothing pruned => store is already current; skip the heavy
        # whole-table re-tag + FTS rebuild (the point of incremental).
        print("tagging profiles onto documents…", flush=True)
        con_lite.execute(
            """
            UPDATE documents SET
              profile = (SELECT json_group_array(dp.profile_id) FROM doc_profiles dp WHERE dp.doc_id = documents.doc_id),
              model  = (SELECT group_concat(DISTINCT p.model)  FROM doc_profiles dp JOIN profiles p ON p.profile_id = dp.profile_id WHERE dp.doc_id = documents.doc_id AND p.model  IS NOT NULL),
              year   = (SELECT group_concat(DISTINCT p.year)   FROM doc_profiles dp JOIN profiles p ON p.profile_id = dp.profile_id WHERE dp.doc_id = documents.doc_id AND p.year   IS NOT NULL),
              engine = (SELECT group_concat(DISTINCT p.engine) FROM doc_profiles dp JOIN profiles p ON p.profile_id = dp.profile_id WHERE dp.doc_id = documents.doc_id AND p.engine IS NOT NULL)
            """
        )
        print("rebuilding FTS…", flush=True)
        con_lite.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        con_lite.commit()
    elif not preview:
        print("store already current — no re-tag / FTS rebuild needed", flush=True)
    con_lite.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="extract only N docs and preview")
    ap.add_argument("--full", action="store_true",
                    help="full rebuild (wipe + re-extract everything); default is incremental")
    args = ap.parse_args()
    extract(limit=args.sample, preview=args.sample is not None, full=args.full)
