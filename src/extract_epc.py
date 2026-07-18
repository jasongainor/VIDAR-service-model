"""Extract EPC parts-catalog into the SQLite store.

Two outputs, both built from one pass over the EPC SQL Server:
  1. one searchable DOCUMENT per Section (TypeId=3) — section title + its part rows;
  2. relational lookup tables `epc_parts` + `epc_section_profiles` that back the
     `epc_part` tool, so it reads the LOCAL store instead of the live SQL Server.
Vehicle applicability comes from ComponentConditions.fkProfile on the section.

Run AFTER extract_docs.py (shares the profiles/doc_profiles tables):
  uv run python src/extract_epc.py                # docs + relational tables (full)
  uv run python src/extract_epc.py --parts-only   # only the relational tables (no FTS rebuild)
"""
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytds

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "vida-kb.sqlite3"
from dbconfig import conn_params  # SQL creds from .env (see .env.example) — never hardcoded
CONN = conn_params()
EN_US, EN_GB = 15, 16


def fetch_epc(cur):
    """Pull (sections, parts_by_section, profiles_by_section) from the EPC SQL Server."""
    cur.execute(
        """
        SELECT cc.Id, cc.FunctionGroupPath,
               coalesce(lus.Description, lgb.Description) AS title
        FROM dbo.CatalogueComponents cc
        LEFT JOIN dbo.Lexicon lus ON lus.DescriptionId = cc.DescriptionId AND lus.fkLanguage = %s
        LEFT JOIN dbo.Lexicon lgb ON lgb.DescriptionId = cc.DescriptionId AND lgb.fkLanguage = %s
        WHERE cc.TypeId = 3
        """,
        (EN_US, EN_GB),
    )
    sections = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    print(f"sections: {len(sections)}", flush=True)

    cur.execute(
        """
        SELECT cc.ParentComponentId, pi.ItemNumber,
               coalesce(lus.Description, lgb.Description), cc.Quantity, cc.TypeId
        FROM dbo.CatalogueComponents cc
        JOIN dbo.PartItems pi ON pi.Id = cc.fkPartItem
        LEFT JOIN dbo.Lexicon lus ON lus.DescriptionId = pi.DescriptionId AND lus.fkLanguage = %s
        LEFT JOIN dbo.Lexicon lgb ON lgb.DescriptionId = pi.DescriptionId AND lgb.fkLanguage = %s
        WHERE cc.TypeId IN (4, 5) AND cc.ParentComponentId IS NOT NULL
        """,
        (EN_US, EN_GB),
    )
    parts_by_section: dict[int, list] = {}
    while True:
        batch = cur.fetchmany(100000)
        if not batch:
            break
        for parent, item, desc, qty, type_id in batch:
            parts_by_section.setdefault(parent, []).append((item, desc, qty, type_id))
    print(f"part rows: {sum(len(v) for v in parts_by_section.values())} under {len(parts_by_section)} parents", flush=True)

    cur.execute("SELECT fkCatalogueComponent, fkProfile FROM dbo.ComponentConditions WHERE fkProfile IS NOT NULL")
    profiles_by_section: dict[int, set] = {}
    while True:
        batch = cur.fetchmany(100000)
        if not batch:
            break
        for comp, prof in batch:
            profiles_by_section.setdefault(comp, set()).add(prof.strip())
    print(f"profile conditions on {len(profiles_by_section)} sections", flush=True)
    return sections, parts_by_section, profiles_by_section


def build_parts_tables(con_lite, sections, parts_by_section, profiles_by_section):
    """Materialize the relational lookup tables that back the epc_part tool locally."""
    con_lite.execute("DROP TABLE IF EXISTS epc_parts")
    con_lite.execute(
        "CREATE TABLE epc_parts (part_number TEXT, description TEXT, function_group TEXT,"
        " section_id INTEGER, section_title TEXT, type_id INTEGER)"
    )
    rows = [
        (item, desc, fg, sec_id, title, type_id)
        for sec_id, (fg, title) in sections.items()
        for (item, desc, _qty, type_id) in parts_by_section.get(sec_id, ())
    ]
    con_lite.executemany("INSERT INTO epc_parts VALUES (?,?,?,?,?,?)", rows)
    con_lite.execute("CREATE INDEX idx_epc_parts_num ON epc_parts(part_number)")
    con_lite.execute("CREATE INDEX idx_epc_parts_desc ON epc_parts(description COLLATE NOCASE)")

    con_lite.execute("DROP TABLE IF EXISTS epc_section_profiles")
    con_lite.execute(
        "CREATE TABLE epc_section_profiles (section_id INTEGER, profile_id TEXT,"
        " PRIMARY KEY (section_id, profile_id))"
    )
    sp = [(sid, p) for sid, profs in profiles_by_section.items() for p in profs]
    con_lite.executemany("INSERT OR IGNORE INTO epc_section_profiles VALUES (?,?)", sp)
    con_lite.execute("CREATE INDEX idx_epc_sp_section ON epc_section_profiles(section_id)")
    con_lite.commit()
    print(f"epc_parts: {len(rows)} rows | epc_section_profiles: {len(sp)} rows", flush=True)


def main():
    t0 = time.time()
    parts_only = "--parts-only" in sys.argv
    con_lite = sqlite3.connect(DB_PATH)
    with pytds.connect(database="EPC", **CONN) as c:
        sections, parts_by_section, profiles_by_section = fetch_epc(c.cursor())

    build_parts_tables(con_lite, sections, parts_by_section, profiles_by_section)
    if parts_only:
        con_lite.close()
        print(f"parts-only done in {time.time()-t0:.0f}s", flush=True)
        return

    con_lite.execute("DELETE FROM documents WHERE db='EPC'")
    con_lite.execute("DELETE FROM doc_profiles WHERE doc_id LIKE 'EPC:%'")
    n = 0
    for sec_id, (fg_path, title) in sections.items():
        parts = parts_by_section.get(sec_id)
        if not parts:
            continue
        doc_id = f"EPC:CatalogueComponents:{sec_id}"
        lines = [f"# Parts: {title or 'section'}", ""]
        if fg_path:
            lines.append(f"_Function group: {fg_path}_\n")
        for item, desc, qty, type_id in parts:
            q = f" (qty {qty})" if qty and str(qty) not in ("1", "1.0") else ""
            ns = " [non-stocked]" if type_id == 5 else ""
            lines.append(f"- {item} — {desc or 'no description'}{q}{ns}")
        con_lite.execute(
            "INSERT OR REPLACE INTO documents (doc_id, db, table_name, source_pk, title,"
            " section, text_md, image_refs) VALUES (?,?,?,?,?,?,?,?)",
            (doc_id, "EPC", "CatalogueComponents", str(sec_id),
             f"Parts: {title or fg_path or sec_id}", "Parts", "\n".join(lines), json.dumps([])),
        )
        for prof in profiles_by_section.get(sec_id, ()):
            con_lite.execute("INSERT OR IGNORE INTO doc_profiles VALUES (?,?)", (doc_id, prof))
        n += 1
    con_lite.commit()
    print(f"EPC docs: {n}", flush=True)

    con_lite.execute(
        """
        UPDATE documents SET
          profile = (SELECT json_group_array(dp.profile_id) FROM doc_profiles dp WHERE dp.doc_id = documents.doc_id),
          model  = (SELECT group_concat(DISTINCT p.model)  FROM doc_profiles dp JOIN profiles p ON p.profile_id = dp.profile_id WHERE dp.doc_id = documents.doc_id AND p.model  IS NOT NULL),
          year   = (SELECT group_concat(DISTINCT p.year)   FROM doc_profiles dp JOIN profiles p ON p.profile_id = dp.profile_id WHERE dp.doc_id = documents.doc_id AND p.year   IS NOT NULL),
          engine = (SELECT group_concat(DISTINCT p.engine) FROM doc_profiles dp JOIN profiles p ON p.profile_id = dp.profile_id WHERE dp.doc_id = documents.doc_id AND p.engine IS NOT NULL)
        WHERE db = 'EPC'
        """
    )
    con_lite.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
    # docs with no real vehicle tagging — surfaced in the car tier, labeled
    con_lite.execute("DROP TABLE IF EXISTS generic_docs")
    con_lite.execute(
        "CREATE TABLE generic_docs AS SELECT d.doc_id FROM documents d"
        " WHERE NOT EXISTS (SELECT 1 FROM doc_profiles dp"
        " JOIN profiles p ON p.profile_id = dp.profile_id"
        " WHERE dp.doc_id = d.doc_id"
        " AND NOT (p.model IS NULL AND p.year IS NULL AND p.engine IS NULL))"
    )
    con_lite.execute("CREATE UNIQUE INDEX idx_generic_docs ON generic_docs(doc_id)")
    con_lite.commit()
    con_lite.close()
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
