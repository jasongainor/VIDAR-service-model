"""dataset/sources.py — read-only DB access over data/vida-kb.sqlite3.

Thin, stdlib-only helpers that hand the example builders (dataset/examples.py)
real rows from the VIDA knowledge base, plus the joins that map a part or a
document to the vehicle(s) it applies to. NOTHING here invents data — every
value, id and vehicle comes straight out of the DB so the generated training
records are grounded in real VIDA content (docs/DECISIONS.md D7).

Schema reference (data/vida-kb.sqlite3 — see docs/BUILD_CONTRACT.md):
  documents(doc_id PK, db, table_name, source_pk, vida_doc_ref, title,
            profile[JSON], model, year, engine, ecu, section, text_md,
            image_refs[JSON])
  epc_parts(part_number, description, function_group, section_id,
            section_title, type_id)
  epc_section_profiles(section_id, profile_id)
  profiles(profile_id, description, model, year, engine, folder_level)
  doc_profiles(doc_id, profile_id)
  pinouts(row_id, module, connector, pin, module_pin, wire_color, goes_to,
          source_pin, goes_to_pin, function, confidence, source_pdf_pages,
          notes, source_file, csv_line)

The document model/year/engine columns are comma-joined lists across platform-mates
("850,S70,V70 (-00)" / "B5204T2,B5254T,...") that are aggregated INDEPENDENTLY, so
their first tokens are not guaranteed to be one real car. _doc_from_row therefore
resolves a coherent (model, year, engine) triple from ONE real profiles row via
doc_profiles->profiles and stashes it on Doc.vehicle; pick_vehicle_for_doc prefers
that, so every synthesized [VEHICLE] block is a car that actually existed (D7).

Pure stdlib (sqlite3, os, json, dataclasses). No third-party imports.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterator, Optional

from grounding import contract as gc

# Default DB path; override with VIDA_KB_DB (relative paths resolve from cwd,
# which the gen CLI runs from the repo root).
DEFAULT_DB = "data/vida-kb.sqlite3"


def db_path() -> str:
    return os.environ.get("VIDA_KB_DB", DEFAULT_DB)


def connect(path: Optional[str] = None) -> sqlite3.Connection:
    """Open the KB read-only with Row access. Raises if the file is absent so
    callers (and the gen self-check) can guard explicitly."""
    p = path or db_path()
    if not os.path.exists(p):
        raise FileNotFoundError(f"VIDA KB not found at {p!r} (set VIDA_KB_DB)")
    # Read-only URI connection: never mutate the corpus from the generator.
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------------- #
# Lightweight row containers
# --------------------------------------------------------------------------- #
@dataclass
class Doc:
    doc_id: str
    vida_doc_ref: Optional[str]
    title: Optional[str]
    model: Optional[str]
    year: Optional[str]
    engine: Optional[str]
    section: Optional[str]
    text_md: Optional[str]
    image_refs: Optional[str]
    # A coherent (model, year, engine) triple from ONE real profiles row this doc
    # applies to (resolved at construction via doc_profiles -> profiles). None when
    # the doc maps to no profile. pick_vehicle_for_doc PREFERS this over the doc's
    # denormalized model/year/engine columns, which are INDEPENDENT comma-lists and
    # must never be zipped into a (model, year, engine) triple that never co-occurred.
    vehicle: Optional["gc.Vehicle"] = None

    def ref(self) -> str:
        """Citation token per D7: VCC ref preferred, doc_id fallback."""
        return self.vida_doc_ref or self.doc_id

    def figure_ids(self) -> list[str]:
        if not self.image_refs:
            return []
        try:
            v = json.loads(self.image_refs)
            return [str(x) for x in v] if isinstance(v, list) else []
        except Exception:
            return []


@dataclass
class Part:
    part_number: str
    description: str
    section_id: Optional[int]
    section_title: Optional[str]


def _coherent_vehicle(conn: sqlite3.Connection, doc_id: str,
                      prefer_model: Optional[str] = None) -> Optional[gc.Vehicle]:
    """ONE real (model, year, engine) profile row the doc actually applies to, via
    doc_profiles -> profiles. Prefers the row whose model == prefer_model (so the
    model stays what the rest of the pipeline expects) but takes THAT row's real
    year+engine — instead of zipping the doc's independently-aggregated model/year/
    engine comma-lists into a triple that may never have existed (the fabrication
    bug this replaces). None when the doc maps to no model-bearing profile."""
    rows = conn.execute(
        "SELECT pr.model, pr.year, pr.engine FROM doc_profiles dp "
        "JOIN profiles pr ON dp.profile_id = pr.profile_id "
        "WHERE dp.doc_id = ? AND pr.model IS NOT NULL "
        "ORDER BY pr.model, pr.year, pr.engine",
        (doc_id,),
    ).fetchall()
    if not rows:
        return None
    chosen = None
    if prefer_model:
        chosen = next((r for r in rows if r["model"] == prefer_model), None)
    chosen = chosen or rows[0]
    return gc.Vehicle(year=chosen["year"], model=chosen["model"],
                      engine=chosen["engine"])


def _doc_from_row(conn: sqlite3.Connection, row: sqlite3.Row) -> Doc:
    doc = Doc(
        doc_id=row["doc_id"],
        vida_doc_ref=row["vida_doc_ref"],
        title=row["title"],
        model=row["model"],
        year=row["year"],
        engine=row["engine"],
        section=row["section"],
        text_md=row["text_md"],
        image_refs=row["image_refs"],
    )
    doc.vehicle = _coherent_vehicle(conn, doc.doc_id,
                                    prefer_model=_first_token(doc.model))
    return doc


_DOC_COLS = ("doc_id, vida_doc_ref, title, model, year, engine, section, "
             "text_md, image_refs")


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def iter_torque_docs(conn: sqlite3.Connection, *, limit: Optional[int] = None
                     ) -> Iterator[Doc]:
    """Documents that carry torque values. We target the ~125 clean
    'Tightening torque' spec sheets first, then any doc with inline 'Tighten to
    ... Nm' prose — torque.py decides what is actually parseable. Ordered
    deterministically (by doc_id) so generation is reproducible."""
    sql = (
        f"SELECT {_DOC_COLS} FROM documents "
        "WHERE (title = 'Tightening torque' OR text_md LIKE '%Tighten to %Nm%') "
        "AND vida_doc_ref IS NOT NULL "
        "ORDER BY (title = 'Tightening torque') DESC, doc_id"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    for row in conn.execute(sql):
        yield _doc_from_row(conn, row)


def docs_for_model(conn: sqlite3.Connection, model: str, *,
                   limit: Optional[int] = None) -> Iterator[Doc]:
    """Documents whose (comma-list) model column mentions `model`. Used to find
    a same-car second topic for multi-turn, and a procedure doc for grounded."""
    sql = (
        f"SELECT {_DOC_COLS} FROM documents "
        "WHERE vida_doc_ref IS NOT NULL AND model LIKE ? "
        "ORDER BY doc_id"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    for row in conn.execute(sql, (f"%{model}%",)):
        yield _doc_from_row(conn, row)


def iter_procedure_docs(conn: sqlite3.Connection, *, limit: Optional[int] = None
                        ) -> Iterator[Doc]:
    """Single-vehicle 'Replacing/Removing/Installing' procedures with body text,
    for grounded procedure examples. model with no comma == one concrete car."""
    sql = (
        f"SELECT {_DOC_COLS} FROM documents "
        "WHERE vida_doc_ref IS NOT NULL AND model IS NOT NULL "
        "AND model NOT LIKE '%,%' AND length(text_md) > 200 "
        "AND (title LIKE 'Replacing%' OR title LIKE 'Removing%' "
        "     OR title LIKE 'Installing%' OR title LIKE 'Checking%') "
        "ORDER BY doc_id"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    for row in conn.execute(sql):
        yield _doc_from_row(conn, row)


def iter_figure_docs(conn: sqlite3.Connection, *, limit: Optional[int] = None
                     ) -> Iterator[Doc]:
    """Documents that actually carry image_refs, for figure-retrieval examples
    (D5: figures = retrieval only)."""
    sql = (
        f"SELECT {_DOC_COLS} FROM documents "
        "WHERE vida_doc_ref IS NOT NULL AND image_refs IS NOT NULL "
        "AND image_refs NOT IN ('', '[]') "
        "ORDER BY doc_id"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    for row in conn.execute(sql):
        yield _doc_from_row(conn, row)


def get_doc(conn: sqlite3.Connection, doc_id: str) -> Optional[Doc]:
    row = conn.execute(
        f"SELECT {_DOC_COLS} FROM documents WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    return _doc_from_row(conn, row) if row else None


# --------------------------------------------------------------------------- #
# Parts
# --------------------------------------------------------------------------- #
def iter_parts(conn: sqlite3.Connection, *, limit: Optional[int] = None
               ) -> Iterator[Part]:
    """Distinct, usable parts: exclude part_number 'ns'/blank and blank
    descriptions, dedupe by part_number (epc_parts has 1.08M rows / 150k
    distinct numbers; the same number recurs across sections)."""
    sql = (
        "SELECT part_number, description, "
        "MIN(section_id) AS section_id, "
        "MIN(section_title) AS section_title "
        "FROM epc_parts "
        "WHERE part_number IS NOT NULL AND part_number != '' "
        "AND part_number != 'ns' "
        "AND description IS NOT NULL AND TRIM(description) != '' "
        "GROUP BY part_number "
        "ORDER BY part_number"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    for row in conn.execute(sql):
        yield Part(
            part_number=row["part_number"],
            description=row["description"],
            section_id=row["section_id"],
            section_title=row["section_title"],
        )


def sibling_vehicle_sharing_engine(conn: sqlite3.Connection, engine: str,
                                   exclude_models: set[str], *, limit: int = 60
                                   ) -> Optional[gc.Vehicle]:
    """A REAL vehicle (from profiles) that carries `engine` but whose model is NOT in
    `exclude_models`. Used to build an HONEST cross-vehicle 'miss': the active car
    genuinely shares the engine yet the part genuinely doesn't list its model. Returns
    None if no such real vehicle exists (so we never fabricate a vehicle/engine pair).

    The match is EXACT (`engine = ?`), not a `LIKE '%engine%'` substring, and the
    returned vehicle is labeled with the engine column from its OWN profile row — so a
    sibling is only emitted when it literally shares the same engine code. (A substring
    match would pull in B5254T3/T7 for a search of B5254T and then mislabel them as
    B5254T, training a shared-engine basis that does not exist — a prime-directive
    violation. See docs/AUDIT-2026-06-20.md.)"""
    if not engine:
        return None
    rows = conn.execute(
        "SELECT DISTINCT model, year, engine FROM profiles "
        "WHERE engine = ? AND model IS NOT NULL AND TRIM(model) != '' "
        "ORDER BY model, year "
        f"LIMIT {int(limit)}",
        (engine,),
    ).fetchall()
    for r in rows:
        m = _first_token(r["model"])
        if m and m not in exclude_models:
            return gc.Vehicle(year=_first_token(r["year"]), model=m, engine=r["engine"])
    return None


def profiles_for_part(conn: sqlite3.Connection, part_number: str, *,
                      limit: int = 25) -> list[tuple[str, str, str]]:
    """(engine, model, year) tuples a part applies to, via
    epc_parts.section_id -> epc_section_profiles -> profiles. Deduped, ordered.
    Empty list when the part's sections map to no resolvable profile."""
    sql = (
        "SELECT DISTINCT pr.engine, pr.model, pr.year "
        "FROM epc_parts p "
        "JOIN epc_section_profiles sp ON p.section_id = sp.section_id "
        "JOIN profiles pr ON sp.profile_id = pr.profile_id "
        "WHERE p.part_number = ? "
        "AND (pr.model IS NOT NULL OR pr.engine IS NOT NULL) "
        "ORDER BY pr.model, pr.year, pr.engine "
        f"LIMIT {int(limit)}"
    )
    out: list[tuple[str, str, str]] = []
    for row in conn.execute(sql, (part_number,)):
        out.append((row["engine"] or "", row["model"] or "", row["year"] or ""))
    return out


# --------------------------------------------------------------------------- #
# Pinouts (one vehicle: 2005 Volvo S60R)
# --------------------------------------------------------------------------- #
S60R_PINOUT_VEHICLE = gc.Vehicle(year="2005", model="S60R", engine="B5254T4")


def iter_pinouts(conn: sqlite3.Connection, *, limit: Optional[int] = None
                 ) -> Iterator[sqlite3.Row]:
    """Pinout rows with a meaningful function description. Citation resolves to
    source_file + csv_line (no vida_doc_ref for pinouts)."""
    sql = (
        "SELECT row_id, module, connector, pin, wire_color, goes_to, function, "
        "source_file, csv_line FROM pinouts "
        "WHERE function IS NOT NULL AND TRIM(function) != '' "
        "ORDER BY row_id"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    yield from conn.execute(sql)


# --------------------------------------------------------------------------- #
# Vehicle synthesis
# --------------------------------------------------------------------------- #
def _first_token(value: Optional[str]) -> Optional[str]:
    """Comma-joined platform lists -> the first concrete value."""
    if not value:
        return None
    head = value.split(",", 1)[0].strip()
    return head or None


def pick_vehicle_for_doc(doc: Doc) -> gc.Vehicle:
    """A concrete grounding.contract.Vehicle for a doc. PREFERS doc.vehicle — ONE
    real profile row resolved at construction (doc_profiles -> profiles), i.e. a
    (model, year, engine) triple that actually co-occurred. Falls back, only when
    the doc has no profile mapping, to the doc's denormalized columns — and even
    then never zips independent comma-lists: a multi-valued year/engine is DROPPED
    rather than paired with an unrelated model, so the [VEHICLE] block is always a
    car that really existed (no fabrication, D7)."""
    if doc.vehicle is not None:
        return doc.vehicle

    def _single(v: Optional[str]) -> Optional[str]:
        return v if (v and "," not in v) else None

    return gc.Vehicle(
        year=_single(doc.year),
        model=_first_token(doc.model),
        engine=_single(doc.engine),
    )


def vehicle_from_profile(engine: str, model: str, year: str) -> gc.Vehicle:
    """Vehicle from a profiles_for_part tuple (same first-token collapse)."""
    return gc.Vehicle(
        year=_first_token(year),
        model=_first_token(model),
        engine=_first_token(engine) or None,
    )


if __name__ == "__main__":
    # Self-check: guard on DB absence so this imports/compiles on a bare Mac.
    p = db_path()
    if not os.path.exists(p):
        print(f"SKIP self-check: KB absent at {p}")
    else:
        c = connect()
        n_torque = sum(1 for _ in iter_torque_docs(c, limit=5))
        n_parts = sum(1 for _ in iter_parts(c, limit=5))
        first_part = next(iter_parts(c, limit=1), None)
        prof = profiles_for_part(c, first_part.part_number)[:2] if first_part else []
        first_doc = next(iter_torque_docs(c, limit=1), None)
        veh = pick_vehicle_for_doc(first_doc) if first_doc else None
        print("torque docs:", n_torque, "| parts:", n_parts)
        if first_part:
            print("part:", first_part.part_number, "-", first_part.description,
                  "| profiles:", prof)
        if veh:
            print("vehicle for first torque doc:", veh.label())
        c.close()
        print("OK")
