"""Create the vida-kb SQLite store and load the pinout CSVs.

Usage:
  uv run python src/build_db.py            # create schema + load pinouts
The documents tables are populated separately by the VIDA extraction pipeline.
"""
import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "vida-kb.sqlite3"
PINOUT_DIR = ROOT / "data" / "pinouts"
ALL_CSV = PINOUT_DIR / "volvo_s60r_2005_ALL_PINOUT_verified.csv"
ECM_CSV = PINOUT_DIR / "volvo_s60r_2005_ECM_pinout_verified.csv"
COMPONENTS_CSV = PINOUT_DIR / "volvo_s60r_2005_components.csv"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id        TEXT PRIMARY KEY,   -- stable, e.g. "DiagSwdl:<table>:<pk>"
  db            TEXT,               -- source database
  table_name    TEXT,
  source_pk     TEXT,               -- primary key in the source table
  vida_doc_ref  TEXT,               -- VIDA's own doc id where present (e.g. D0088)
  title         TEXT,
  profile       TEXT,               -- VIDA vehicle profile id(s), JSON array
  model TEXT, year TEXT, engine TEXT, ecu TEXT, section TEXT,
  text_md       TEXT,               -- normalized searchable text
  image_refs    TEXT                -- JSON array of image ids
);
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
  title, text_md, content='documents', content_rowid='rowid'
);

CREATE TABLE IF NOT EXISTS raw_documents (   -- kept for later PDF rendering
  doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id),
  xml TEXT, image_refs TEXT
);

CREATE TABLE IF NOT EXISTS pinouts (
  row_id INTEGER PRIMARY KEY,
  module TEXT, connector TEXT, pin TEXT, module_pin TEXT,
  wire_color TEXT, goes_to TEXT, source_pin TEXT, goes_to_pin TEXT,
  function TEXT, confidence TEXT, source_pdf_pages TEXT, notes TEXT,
  source_file TEXT, csv_line INTEGER          -- citation resolves to file+line
);
CREATE VIRTUAL TABLE IF NOT EXISTS pinouts_fts USING fts5(
  module, pin, function, goes_to, notes, component, tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS components (
  component_id TEXT PRIMARY KEY,
  description TEXT,
  source_file TEXT, csv_line INTEGER
);
"""


def load_pinouts(con: sqlite3.Connection) -> None:
    con.execute("DELETE FROM pinouts")
    con.execute("DELETE FROM components")

    with open(ALL_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows, start=2):  # line 1 is the header
        con.execute(
            "INSERT INTO pinouts (module, connector, pin, module_pin, wire_color,"
            " goes_to, source_pin, goes_to_pin, function, confidence,"
            " source_pdf_pages, notes, source_file, csv_line)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                r["Module"], r["Connector"], r["Pin"], r.get("Module Pin"),
                r["Wire Color"], r["Goes To"], r["Source Pin"], r.get("Goes To Pin"),
                r["Function / Role"], r["Confidence"], r["Source PDF Pages"],
                r["Notes"], ALL_CSV.name, i,
            ),
        )

    # The ECM-only CSV should be a subset of the ALL file; verify instead of
    # double-loading so each fact has exactly one citation row.
    with open(ECM_CSV, newline="", encoding="utf-8-sig") as f:
        ecm = list(csv.DictReader(f))
    missing = []
    for r in ecm:
        hit = con.execute(
            "SELECT 1 FROM pinouts WHERE module=? AND connector=? AND pin=? AND wire_color=?",
            (r["Module"], r["Connector"], r["Pin"], r["Wire Color"]),
        ).fetchone()
        if not hit:
            missing.append((r["Module"], r["Connector"], r["Pin"]))
    if missing:
        print(f"WARNING: {len(missing)} ECM rows not covered by ALL csv: {missing[:5]}", file=sys.stderr)

    with open(COMPONENTS_CSV, newline="", encoding="utf-8-sig") as f:
        comps = list(csv.DictReader(f))
    for i, r in enumerate(comps, start=2):
        con.execute(
            "INSERT INTO components (component_id, description, source_file, csv_line) VALUES (?,?,?,?)",
            (r["Component ID"], r["Description"], COMPONENTS_CSV.name, i),
        )

    con.execute("DELETE FROM pinouts_fts")
    con.execute(
        "INSERT INTO pinouts_fts(rowid, module, pin, function, goes_to, notes, component)"
        " SELECT p.row_id, p.module, p.pin, p.function, p.goes_to, p.notes,"
        " coalesce(c.description, '') FROM pinouts p"
        " LEFT JOIN components c ON c.component_id = p.goes_to"
    )
    n_pin = con.execute("SELECT COUNT(*) FROM pinouts").fetchone()[0]
    n_comp = con.execute("SELECT COUNT(*) FROM components").fetchone()[0]
    print(f"loaded {n_pin} pinout rows, {n_comp} components (ECM subset check: {len(missing)} missing)")


def main():
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    load_pinouts(con)
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
