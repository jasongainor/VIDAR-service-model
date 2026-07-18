"""vida-kb MCP server — retrieval over the single SQLite store.

Run:
  uv run python src/server.py                  # stdio (Claude Desktop, OpenCode)
  uv run python src/server.py --http [port]    # streamable HTTP (Open WebUI / mcpo)

Every result carries a citation that resolves back to the source row
(DB table + pk for VIDA docs, CSV file + line for pinouts).
"""
import json
import logging
import math
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Image

# pytds logs every login/RPC at INFO; we open a fresh read-only connection per
# SQL-backed call (epc_part, get_figures), so keep it to warnings and above.
logging.getLogger("pytds").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "vida-kb.sqlite3"

# ── Vehicle context ──────────────────────────────────────────────────────────
# Everything car-specific lives in ONE object so the tool is not nailed to a single
# vehicle: retrieval scoping, EPC applicability, the engine-family safety gates and the
# user-facing labels all read from the active CarContext. It is resolved once at import,
# lowest→highest priority: the baked-in S60R default, an optional car.toml at the repo
# root, then VIDA_CAR_* env vars. So this build needs zero config, and pointing the tool
# at a different VIDA car is a one-file edit (see car.example.toml) — no code changes.
# This is the foundation for the all-cars goal; a later step can make the context a
# per-query parameter rather than per-deployment.
#
# Engine applicability is the safety-critical part: torque/clearance specs and the
# figures that depict them are ENGINE-SPECIFIC, so retrieval must never cross engine
# families (a 6-cylinder diesel head-bolt diagram is actively dangerous on a 5-cylinder
# petrol). VIDA encodes engine identity two ways — document TAGGING (doc_profiles →
# profiles.engine), and engine codes named in surrounding TEXT — and both are gated.
# "Compatible" = engines sharing this car's head/torque geometry, i.e. its engine family
# (`engine_family` regex; for B5254T4 that is the B5xxx white-block 5-cyl petrol family).
@dataclass(frozen=True)
class CarContext:
    profile_id: str          # VIDA vehicle-profile id (EPC ComponentConditions.fkProfile)
    engine: str              # exact engine code, e.g. "B5254T4"
    model: str | None        # VIDA profiles.model, e.g. "S60 (-09)" (None = match any)
    year: str | None         # VIDA profiles.year, e.g. "2005" (None = match any)
    label: str               # human description, e.g. "2005 Volvo S60R (B5254T4)"
    engine_family: str       # regex (text) for engines sharing this car's head/torque
                             # geometry; defaults to <block+cyl-digit>\d{3} from `engine`
    aliases: tuple[str, ...]  # short tokens that mean "this car" in search_docs(profile=…)
    pinout_doc: str          # short wiring-source ref used in pin citations
    pinout_source: str       # full wiring-source description for list_sources


def _default_engine_family(engine: str) -> str:
    """Engines sharing this car's head/torque geometry = same block letter + cylinder
    digit (B5xxx for B5254T4, B4xxx for B4204T, D5xxx for D5244T). Volvo modern/diesel
    codes put the cylinder count in the first digit; 3 displacement digits follow."""
    m = re.match(r"([A-Za-z]+\d)", engine)
    return rf"{m.group(1) if m else engine[:2]}\d{{3}}"


# The baked-in default car (this build's actual vehicle). `aliases` is kept OUT of the
# field defaults so a partial override (engine changed, aliases not) can't inherit the
# default car's nickname — the engine code is ALWAYS an alias, and these extra nicknames
# attach only when the engine is still the default's.
_DEFAULTS = dict(
    profile_id="0b00c8af815cadc9", engine="B5254T4", model="S60 (-09)", year="2005",
    label="2005 Volvo S60R (B5254T4)", pinout_doc="TP 3976201",
    pinout_source="TP 3976201 wiring diagrams via community-verified CSV",
)
_DEFAULT_NICKS = ("s60r",)


def _load_car_context() -> CarContext:
    cfg = dict(_DEFAULTS)
    toml_path = ROOT / "car.toml"
    if toml_path.exists():
        try:
            import tomllib
            with open(toml_path, "rb") as fh:
                cfg.update({k: v for k, v in tomllib.load(fh).get("car", {}).items()
                            if v is not None})
        except Exception as e:  # malformed config must never take the tool down
            print(f"car.toml ignored ({e}); using defaults", file=sys.stderr)
    for key in ("profile_id", "engine", "model", "year", "label", "engine_family",
                "aliases", "pinout_doc", "pinout_source"):
        ev = os.environ.get(f"VIDA_CAR_{key.upper()}")
        if ev:
            cfg[key] = ev
    eng = cfg["engine"]
    # aliases: the engine code is always one; plus explicit extras (toml list or env
    # comma/space string), else the default car's nicknames when it IS the default car.
    extras = cfg.get("aliases")
    if extras is None:
        extras = _DEFAULT_NICKS if eng == _DEFAULTS["engine"] else ()
    elif isinstance(extras, str):
        extras = [a for a in re.split(r"[,\s]+", extras) if a]
    aliases = tuple(dict.fromkeys([eng.lower(), *(a.lower() for a in extras)]))
    return CarContext(
        profile_id=cfg["profile_id"], engine=eng, model=cfg.get("model"),
        year=cfg.get("year"), label=cfg.get("label") or eng,
        engine_family=cfg.get("engine_family") or _default_engine_family(eng),
        aliases=aliases,
        pinout_doc=cfg.get("pinout_doc") or "", pinout_source=cfg.get("pinout_source") or "",
    )


CAR = _load_car_context()
_COMPAT_ENGINE_RE = re.compile(CAR.engine_family, re.I)  # compiled once for the hot path
# Volvo ENGINE codes. The numeric CORE encodes cylinders+displacement and is short
# (red-block B2xx = 3 digits, modern B?xxx = 4, diesel D + 2–4); any variant suffix
# that follows starts with a LETTER (T4, FT, TIC, S2). The trailing (?!\d) after each
# core is load-bearing twice over: it rejects body DTCs (B2155) AND long numeric
# part/tool/document ids that begin with the same letter (D5200235, B5200401) — those
# are NOT engines and must never trip the wrong-engine gate.
#   red-block:  B2 + 2 digits           (B230, B230FT)
#   modern:     B{4,5,6,8} + 3 digits   (B5254T4, B6304T, B4204T2)
#   diesel:     D + 2–4 digits          (D24, D24TIC, D5244T18)
_ENGINE_FAMILY_RE = re.compile(
    r"\bB2\d{2}(?!\d)(?:[A-Z][A-Z0-9]*)?"
    r"|\bB[4568]\d{3}(?!\d)(?:[A-Z][A-Z0-9]*)?"
    r"|\bD\d{2,4}(?!\d)(?:[A-Z][A-Z0-9]*)?"
)


def _resolve_car(spec: str) -> CarContext:
    """Build a CarContext for a per-QUERY car override (item 7): the deployment car is the
    default, but search_docs(car=…) can scope a single query to a DIFFERENT vehicle in the
    same VIDA install (the all-cars goal, one query at a time). `spec` is a short string —
    an engine code ('B6294T', 'D5244T'), optionally with a model and/or year token
    ('XC90 B6294T', 'B4204T 2016'). The engine code drives the safety-critical family
    regex; a year token (if present) tightens document scoping. Falls back to the module
    CAR when no engine code is found, so a vague spec can never silently widen scope."""
    s = (spec or "").strip()
    eng = next((tok.upper() for tok in re.findall(r"[A-Za-z0-9]+", s)
                if _ENGINE_FAMILY_RE.fullmatch(tok)), None)
    if not eng:
        return CAR
    ym = re.search(r"\b((?:19|20)\d{2})\b", s)
    return CarContext(
        profile_id="", engine=eng, model=None, year=(ym.group(1) if ym else None),
        label=f"{s} (engine {eng})", engine_family=_default_engine_family(eng),
        aliases=(eng.lower(),), pinout_doc=CAR.pinout_doc, pinout_source=CAR.pinout_source,
    )


def _engine_compatible(code: str, compat_re=None) -> bool:
    """True iff a Volvo engine code shares THIS car's head/torque geometry — its engine
    family (CAR.engine_family; for B5254T4 the B5xxx white-block 5-cyl petrol family).
    Other cylinder counts, blocks and fuel types do not and must never be cross-served.
    compat_re overrides the module car's family regex for a per-query car (item 7)."""
    return bool((compat_re or _COMPAT_ENGINE_RE).match(code))


def _wrong_engine_near(text: str, pos: int, window: int = 280) -> bool:
    """True if the engine code NEAREST this figure's marker is for a DIFFERENT engine.
    Judging by the closest code (not 'any code in a wide window') is essential in
    multi-engine torque tables, where a B5254 subsection sits a few lines from a
    B4204 one — the figure belongs to whichever engine its own caption names."""
    seg0 = max(0, pos - window)
    seg = (text or "")[seg0: pos + window]
    nearest = None
    for mo in _ENGINE_FAMILY_RE.finditer(seg):
        d = abs((seg0 + mo.start()) - pos)
        if nearest is None or d < nearest[0]:
            nearest = (d, _engine_compatible(mo.group(0)))
    return nearest is not None and not nearest[1]


def _engine_token_sql(col: str) -> str:
    """SQL predicate: the comma-joined engine column lists this car's EXACT engine.
    Comma-tokenized (wrap+strip-spaces) so a stem like 'B5254T' can't match 'B5254T4'."""
    return f"(',' || REPLACE(IFNULL({col}, ''), ' ', '') || ',') LIKE '%,{CAR.engine},%'"


def _engine_applicability(engine: str | None, compat_re=None) -> str:
    """Classify a document's denormalized engine list (documents.engine) for THIS car.
      'compatible' — lists this car's white-block 5-cyl family (B5xxx); safe to quote.
      'wrong'      — engine-specific to a DIFFERENT family ONLY (no compatible engine in
                     the list); its torque/clearance/spec values are unsafe for this car.
      'generic'    — no recognized engine tagging (shared/boilerplate); safe.
    A doc listing BOTH a compatible and an incompatible engine (multi-engine torque
    tables) is 'compatible' — the correct value is present and the doc must be readable.
    compat_re overrides the module car's family regex for a per-query car (item 7)."""
    toks = [t for t in re.sub(r"\s+", "", engine or "").split(",") if t]
    if not toks:
        return "generic"
    if any(_engine_compatible(t, compat_re) for t in toks):
        return "compatible"
    if any(_ENGINE_FAMILY_RE.match(t) for t in toks):  # a recognized incompatible code
        return "wrong"
    return "generic"  # only unrecognized/vintage codes — not a known hazard family


def _lead_wrong_engine(text: str, compat_re=None) -> bool:
    """True if a lead/snippet names ONLY incompatible engine codes (no compatible
    B5xxx anywhere in it). This is the content-aware backstop for UNTAGGED docs whose
    engine column is empty but whose prose is actually about a different engine — the
    same hole that, on figures, leaked a B4204 illustration from a 'generic' doc."""
    codes = _ENGINE_FAMILY_RE.findall(text or "")
    if not codes or any(_engine_compatible(c, compat_re) for c in codes):
        return False
    return True

mcp = FastMCP("vida-kb")


def _tool(*dargs, **dkwargs):
    """@mcp.tool() that fills {car}/{engine}/{aliases} placeholders in the docstring
    from the active CarContext BEFORE FastMCP captures it as the model-facing tool
    description — so the tool guidance names the configured car, not a hardcoded S60R.
    No-op for docstrings without placeholders; falls back to the raw text if a stray
    brace makes formatting fail, so a docstring edit can never break import."""
    def deco(fn):
        if fn.__doc__:
            try:
                fn.__doc__ = fn.__doc__.format(
                    car=CAR.label, engine=CAR.engine, pinout=CAR.pinout_doc,
                    aliases=", ".join(CAR.aliases))
            except (KeyError, IndexError, ValueError):
                pass
        return mcp.tool(*dargs, **dkwargs)(fn)
    return deco


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _env(key: str) -> str | None:
    """Read a value from process env, falling back to the gitignored .env file
    (VIDA_SQL_USER / VIDA_SQL_PASSWORD for the read-only SQL login)."""
    if key in os.environ:
        return os.environ[key]
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return None


def _sql(database: str):
    """Open a READ-ONLY pytds connection (vida_ro login) to one VIDA SQL Server
    database. Returns None if the credentials or the server are unavailable, so
    the SQL-backed tools can degrade gracefully instead of erroring the turn.
    The login has only db_datareader; the server itself rejects any write."""
    user, pw = _env("VIDA_SQL_USER"), _env("VIDA_SQL_PASSWORD")
    if not user or not pw:
        return None
    try:
        import pytds

        return pytds.connect(
            server="127.0.0.1", port=1433, database=database, user=user,
            password=pw, autocommit=True, login_timeout=3, timeout=20,
        )
    except Exception:
        return None


# Mechanic vocabulary -> VIDA/Volvo-British vocabulary. Each token expands to an
# OR-group, so this only ever ADDS recall. Keep entries token-level (FTS terms).
_SYN = {
    "headlight": ["headlamp"], "headlights": ["headlamps"],
    "aiming": ["aligning", "alignment"], "aim": ["align"],
    "windshield": ["windscreen"], "windshields": ["windscreens"],
    "serpentine": ["auxiliaries", "auxiliary", "drive"],
    "intake": ["inlet"], "inlet": ["intake"],
    "rail": ["pipe", "pipes"],
    "capacity": ["volume", "liters", "litres"],
    "oxygen": ["o2", "ho2s"],
    "reset": ["initializing", "initialization", "calibration"],
    "calibration": ["initializing", "adaptation"], "calibrate": ["initialize", "adapt"],
    "boot": ["trunk"], "bonnet": ["hood"], "fender": ["wing"],
    "muffler": ["silencer"], "abs": ["bcm", "brake"],
    "gap": ["clearance", "electrode"],
    # VIDA/Volvo-British calls a bolt a "screw" — the this-car torque tables say
    # "Transmission screw", "M6x20 … Allen", never "bolt". Without this bridge a query
    # phrased with "bolt(s)" can't strictly match the authoritative spec table.
    "bolt": ["screw"], "bolts": ["screws"], "screw": ["bolt"], "screws": ["bolts"],
    # VIDA tightening procedures say "sequence", never "pattern" — bridge it. Map to
    # "sequence" only (NOT the ubiquitous "order", which fires the figure gate on
    # unrelated docs and lets an ambiguous bare "head …" match a wrong figure).
    "pattern": ["sequence"], "patterns": ["sequences"],
}
# Context-dependent expansions: when BOTH words appear, the second gains alternatives.
_BIGRAM_SYN = {
    ("throttle", "body"): ("body", ["unit", "module"]),
    ("valve", "cover"): ("valve", ["camshaft"]),
}
# Tokens that never appear in the corpus phrasing and only dilute matching.
_STOP = {"part", "parts", "number", "oem", "pn",
         "procedure", "procedures", "instructions", "instruction", "steps",
         "guide", "how", "howto", "recommended", "proper", "correct", "official"}


def _load_db_synonyms() -> dict[str, list[str]]:
    """Synonyms mined at ingestion from VIDA's own en-US/en-GB Lexicon (build_synonyms.py),
    merged on top of the small hand-curated _SYN. Best-effort: empty if not built yet."""
    out: dict[str, list[str]] = {}
    try:
        with _con() as con:
            for term, alt in con.execute(
                "SELECT term, alt FROM synonyms ORDER BY n DESC"
            ):
                out.setdefault(term, []).append(alt)
    except sqlite3.OperationalError:
        pass  # synonyms table not built — hand-curated _SYN still applies
    return out


_DB_SYN = _load_db_synonyms()


def _load_db_phrases() -> dict[str, list[str]]:
    """Multi-word / unequal-length phrase synonyms (item 10 v2), mined by build_synonyms.py
    from the en-US/en-GB Lexicon ('gear box'↔'gearbox', 'anti roll bar'↔'sway bar') — the
    variants the single-word minimal-pair miner cannot represent. Empty if not built."""
    out: dict[str, list[str]] = {}
    try:
        with _con() as con:
            for phrase, alt in con.execute("SELECT phrase, alt FROM phrase_synonyms ORDER BY n DESC"):
                out.setdefault(phrase, []).append(alt)
    except sqlite3.OperationalError:
        pass
    return out


_DB_PHRASE = _load_db_phrases()


def _syn_alts(term: str) -> list[str]:
    """Merged synonym alternatives for a query term: hand-curated first, then mined,
    de-duped and capped so a single term can't explode the FTS OR-group."""
    merged = list(dict.fromkeys([*_SYN.get(term, []), *_DB_SYN.get(term, [])]))
    return merged[:8]


def _phrase_alts(terms: list[str]) -> list[tuple[str, str]]:
    """(last_token, alt_phrase) for every known source phrase that appears as a CONTIGUOUS
    run in the query tokens. Attaching the alt to the source phrase's last token folds it
    into the existing OR-group machinery — pure recall, never disturbs strict ranking."""
    if not _DB_PHRASE:
        return []
    n = len(terms)
    found = []
    for phrase, alts in _DB_PHRASE.items():
        pw = phrase.split()
        if not (2 <= len(pw) <= n):
            continue
        if any(terms[i:i + len(pw)] == pw for i in range(n - len(pw) + 1)):
            found.extend((pw[-1], alt) for alt in alts)
    return found


def _terms(query: str) -> list[str]:
    terms = [t.lower() for t in re.findall(r"[A-Za-z0-9_.:/-]+", query)]
    kept = [t for t in terms if t not in _STOP]
    return kept or terms


def _fts_match(query: str, relaxed: bool = False) -> str:
    """Build an FTS5 MATCH string: AND of synonym OR-groups (or flat OR when relaxed)."""
    terms = _terms(query)
    if not terms:
        return '""'
    alts_by_term = {t: [t] + _syn_alts(t) for t in terms}
    for (a, b), (target, extra) in _BIGRAM_SYN.items():
        if a in alts_by_term and b in alts_by_term:
            alts_by_term[target] = alts_by_term[target] + extra
    for last, altphrase in _phrase_alts(terms):  # item 10 v2: multi-word phrase recall
        if last in alts_by_term and altphrase not in alts_by_term[last]:
            alts_by_term[last] = alts_by_term[last] + [altphrase]
    if relaxed:
        flat = dict.fromkeys(alt for alts in alts_by_term.values() for alt in alts)
        return " OR ".join(f'"{t}"' for t in flat)
    return " AND ".join(
        "(" + " OR ".join(f'"{a}"' for a in alts) + ")" for alts in alts_by_term.values()
    )


def _fts_quote(query: str) -> str:  # backwards-compatible name
    return _fts_match(query)


def _short(val: str | None, n: int = 160) -> str | None:
    if val and len(val) > n:
        kept = val[:n].rsplit(",", 1)[0]
        return f"{kept} (+{val.count(',') - kept.count(',')} more)"
    return val


def _doc_citation(row: sqlite3.Row) -> str:
    scope = _short(row["engine"], 60) or "all profiles"
    label = f"{row['title']} — {scope} ({row['db']}/{row['table_name']}#{row['source_pk']})"
    if row["vida_doc_ref"]:
        label += f" [{row['vida_doc_ref']}]"
    return label


def _pin_citation(row: sqlite3.Row) -> str:
    cite = (
        f"{row['module']} {row['connector']}:{row['pin']} — {CAR.pinout_doc} "
        f"p.{row['source_pdf_pages']}, confidence {row['confidence']}"
    )
    if row["notes"]:
        cite += f"; {row['notes']}"
    return cite


# Documents applicable to THE car per VIDA's own DocumentProfile mapping: the car's
# exact profile, its ancestors (model-only, model+year), and engine-matched variants.
# Built from the active CarContext (literals escaped — values are deployer config, not
# user input). Exact-engine docs count regardless of tagged year (VIDA often tags a
# shared engine procedure only to a later year); a year-only profile must still match
# this car's year — a facelift makes cross-year body docs unsafe.
def _sql_lit(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _build_car_docs_sql(car: CarContext) -> str:
    eng = _sql_lit(car.engine)
    model_pred = f"(model IS NULL OR model = {_sql_lit(car.model)})" if car.model else "1=1"
    year_pred = (f"((year IS NULL OR year = {_sql_lit(car.year)}) OR engine = {eng})"
                 if car.year else "1=1")
    return (
        "SELECT doc_id FROM doc_profiles WHERE profile_id IN ("
        " SELECT profile_id FROM profiles"
        f" WHERE {model_pred}"
        f" AND {year_pred}"
        f" AND (engine IS NULL OR engine = {eng})"
        " AND NOT (model IS NULL AND year IS NULL AND engine IS NULL))"
    )


_CAR_DOCS_SQL = _build_car_docs_sql(CAR)
# Docs with no real vehicle tagging at all (generic content: Four-C, terminal
# repair, tools…) — they belong in the car tier, labeled, not buried.
_TIER1_SQL = f"{_CAR_DOCS_SQL} UNION SELECT doc_id FROM generic_docs"


# ── Hybrid retrieval (FTS5 + dense vectors) ──────────────────────────────────
# build_embeddings.py embeds every doc with a LOCAL model into data/vida-vectors.sqlite3
# (sqlite-vec). search_docs fuses FTS5/BM25 (exact tokens — part numbers, torque values,
# VCC refs) with vector KNN (semantic recall — "make the AWD quieter" -> angle-gear docs)
# via Reciprocal Rank Fusion. Best-effort: if the vectors or the embedding endpoint are
# absent/locked, search_docs degrades to pure FTS. Engine/scope enforcement runs on the
# fused set, so the safety guarantees are unchanged.
VECTORS_DB = ROOT / "data" / "vida-vectors.sqlite3"
EMBED_URL = _env("VIDA_EMBED_URL") or "http://localhost:1234/v1"
EMBED_MODEL = _env("VIDA_EMBED_MODEL") or "text-embedding-nomic-embed-text-v1.5"
# Kill switch for the dense layer: VIDA_DISABLE_VECTORS=1 forces pure-FTS retrieval.
# Used by the retrieval scorecard to isolate lexical-ranking changes from vector flap,
# and as a graceful escape hatch if the embedding endpoint is down/misbehaving.
_VECTORS_ENABLED = (_env("VIDA_DISABLE_VECTORS") or "").lower() not in ("1", "true", "yes")
# Vector-injection precision gate (item 3). nomic L2 distances on this corpus run ~0.6
# (very on-topic) to ~1.0 (vague). A vector-only doc is injected only when it is BOTH
# below the absolute ceiling AND within the relative margin of the best vector hit; docs
# nearer than _VEC_TIGHT bypass the lexical-overlap guardrail entirely. Env-tunable so the
# scorecard can calibrate without code edits.
_VEC_ABS_CEIL = float(_env("VIDA_VEC_ABS_CEIL") or "0.92")
_VEC_REL_MARGIN = float(_env("VIDA_VEC_REL_MARGIN") or "0.16")
_VEC_TIGHT = float(_env("VIDA_VEC_TIGHT") or "0.74")
# Weighted reciprocal-rank fusion (item 1). The invariant that earlier blocked RRF —
# "never demote a strong exact match" — is kept by LOCKING strict lexical hits above the
# fused pool (they are not re-ranked at all). RRF only orders the LOW-confidence pool:
# loose relaxed-OR lexical hits (weight _RRF_W_LOOSE) vs gated vector hits (_RRF_W_VEC).
# So on a query with no strict lexical match ("key won't turn"), a confident vector hit
# ("Key warning") outranks loose lexical junk ("Ski holder"); on a strict query the lock
# makes fusion behave exactly like the old additive fill. Env-tunable for the scorecard.
_RRF_K = float(_env("VIDA_RRF_K") or "60")
_RRF_W_LOOSE = float(_env("VIDA_RRF_W_LOOSE") or "0.30")
_RRF_W_VEC = float(_env("VIDA_RRF_W_VEC") or "1.0")


def _embed_query(query: str):
    """Embed a query with the local endpoint (nomic 'search_query:' prefix), or None."""
    try:
        import httpx
        r = httpx.post(f"{EMBED_URL}/embeddings",
                       json={"model": EMBED_MODEL, "input": [f"search_query: {query}"]},
                       timeout=12)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
    except Exception:
        return None


def _vector_doc_ids(query: str, k: int = 40, with_distance: bool = False):
    """Top-k doc_ids by semantic similarity (sqlite-vec KNN), nearest first. Returns a
    list of doc_ids, or (doc_id, distance) pairs when with_distance=True. [] if the
    vectors/endpoint are unavailable or the store is momentarily writer-locked — the
    caller falls back to FTS."""
    if not VECTORS_DB.exists():
        return []
    emb = _embed_query(query)
    if not emb:
        return []
    con = None
    try:
        import struct
        import sqlite_vec
        con = sqlite3.connect(f"file:{VECTORS_DB}?mode=ro", uri=True, timeout=2)
        con.execute("PRAGMA busy_timeout=2000")
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        blob = struct.pack(f"{len(emb)}f", *emb)
        hits = con.execute(
            "SELECT rowid, distance FROM vec_docs WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (blob, k)).fetchall()
        if not hits:
            return []
        rowids = [h[0] for h in hits]
        qm = ",".join("?" * len(rowids))
        m = dict(con.execute(f"SELECT rowid, doc_id FROM vec_map WHERE rowid IN ({qm})", rowids))
        out = [(m[rid], dist) for rid, dist in hits if rid in m]
        return out if with_distance else [d for d, _ in out]
    except Exception:
        return []
    finally:
        if con is not None:
            con.close()


def _scope_for_ids(con, doc_ids: list[str], car_docs_sql: str | None = None) -> dict[str, str]:
    """Label each doc_id 'this car' / 'not vehicle-specific' / 'other vehicles' the way the
    FTS tier does — for vector-added docs that bypassed the tier logic. car_docs_sql lets a
    per-query car (item 7) label against that car's applicability instead of the default."""
    if not doc_ids:
        return {}
    qm = ",".join("?" * len(doc_ids))
    carset = {r[0] for r in con.execute(
        f"SELECT doc_id FROM ({car_docs_sql or _CAR_DOCS_SQL}) WHERE doc_id IN ({qm})", doc_ids)}
    genset = {r[0] for r in con.execute(
        f"SELECT doc_id FROM generic_docs WHERE doc_id IN ({qm})", doc_ids)}
    return {d: ("not vehicle-specific" if d in genset
               else "this car" if d in carset else "other vehicles") for d in doc_ids}


_FUSE_COLS = ("doc_id, title, profile, model, year, engine, section, db, table_name,"
              " source_pk, vida_doc_ref, substr(text_md,1,200) AS snip, 0 AS rank")


def _augment_with_vectors(con, query: str, rows, scopes: dict, lim: int, low_conf: bool = False,
                          car_docs_sql: str | None = None):
    """Blend dense (vector) recall into the lexical/tiered FTS result. Two regimes,
    switched by whether the lexical search found ANY strict (all-terms) match:

    • Normal (low_conf=False) — there were strict lexical matches, so the lexical order is
      trustworthy: preserve it EXACTLY and let gated vector hits fill only the remaining
      slots (the original additive behavior; a strong exact match is never displaced).
      Keeps 'Checking clutch shudder' on top for 'shuddering when accelerating'.

    • Low-confidence (low_conf=True) — the lexical side is all loose relaxed-OR matches
      (no strict hit anywhere), i.e. likely junk ('Ski holder' for 'key won't turn'). Here
      a weighted RRF (item 1) lets a CONFIDENT vector hit ('Key warning') outrank the loose
      lexical noise; loose lexical carries the small _RRF_W_LOOSE weight, vectors _RRF_W_VEC.

    Both regimes pass vector candidates through the precision gate (item 3) first, and
    engine applicability is still enforced downstream in _format_results."""
    rows = list(rows)
    have = {r["doc_id"] for r in rows}

    # Gated vector candidates (item 3 precision gate): a vector doc earns a place only when
    # the embedding is CONFIDENT — below an absolute distance ceiling AND within a relative
    # margin of the best hit. (A flat threshold can't separate signal from noise on this
    # corpus; relative-to-best can.) vec_order also records the rank of docs already in the
    # lexical pool, so RRF can reward lexical+vector agreement.
    vec_order: dict[str, int] = {}
    vec_dist: dict[str, float] = {}
    need_ids: list[str] = []
    if _VECTORS_ENABLED:
        ranked = _vector_doc_ids(query, with_distance=True) or []
        best = ranked[0][1] if ranked else 0.0
        for doc_id, dist in ranked:
            vec_order.setdefault(doc_id, len(vec_order))
            vec_dist[doc_id] = dist
            if doc_id in have:
                continue
            if dist <= _VEC_ABS_CEIL and (dist - best) <= _VEC_REL_MARGIN:
                need_ids.append(doc_id)

    # Fetch the vector-only candidates and apply the lexical-overlap guardrail: a
    # vector-only doc must share a query content term (or synonym) UNLESS it is very close
    # in embedding space — the final junk filter (drops "ski holder", keeps "clutch shudder").
    fetched = []
    if need_ids:
        qset = {t for t in _terms(query) if len(t) >= 3}
        qset |= {s for t in list(qset) for s in _syn_alts(t)}
        qm = ",".join("?" * len(need_ids))
        for r in con.execute(f"SELECT {_FUSE_COLS} FROM documents WHERE doc_id IN ({qm})", need_ids):
            if vec_dist.get(r["doc_id"], 1.0) <= _VEC_TIGHT or _title_or_terms_overlap(r, qset):
                fetched.append(r)

    if not low_conf:
        room = max(0, lim - len(rows))
        added = sorted(fetched, key=lambda r: vec_order.get(r["doc_id"], 1 << 30))[:room]
        out = (rows + added)[:lim]
    else:
        lex_rank = {r["doc_id"]: i for i, r in enumerate(rows)}

        def rrf(r):
            did = r["doc_id"]
            s = 0.0
            if did in lex_rank:
                s += _RRF_W_LOOSE / (_RRF_K + lex_rank[did])
            if did in vec_order:
                s += _RRF_W_VEC / (_RRF_K + vec_order[did])
            return s

        seen, out = set(), []
        for r in sorted(rows + fetched, key=rrf, reverse=True):
            if r["doc_id"] not in seen:
                seen.add(r["doc_id"])
                out.append(r)
        out = out[:lim]
    scopes = {**_scope_for_ids(con, [r["doc_id"] for r in fetched], car_docs_sql), **scopes}
    return out, scopes


def _title_or_terms_overlap(row, qset: set[str]) -> bool:
    """True if the doc's title or snippet shares any query content term/synonym — the
    lexical anchor a confident-but-vague vector hit needs to earn a result slot."""
    hay = f"{row['title'] or ''} {row['snip'] or ''}".lower()
    return any(t in hay for t in qset)


# ── Lexical re-rank (item 2) ─────────────────────────────────────────────────
# BM25 alone mis-ranks when a doc merely CONTAINS the query words: "Intake manifold,
# replacement" used to beat the cylinder-block doc for "cylinder head bolt torque", and
# the authoritative this-car spec table ("Tightening torque", value in a table ROW, not
# the title) sank below docs whose title literally said "intake manifold". Fix: fetch a
# wider candidate set, then re-rank with two domain priors — (1) reward query terms that
# appear in the TITLE (a title match is a stronger signal than a body mention), and (2)
# on a torque-intent query, boost the dedicated "Tightening torque(s)" spec tables, which
# carry the verified values. Lower score = better, matching the bm25 `rank` convention.
_CAND = int(_env("VIDA_RERANK_CAND") or "30")       # candidates fetched per tier before rerank
_TITLE_TERM_BONUS = float(_env("VIDA_TITLE_BONUS") or "2.0")
_SPEC_TABLE_BONUS = float(_env("VIDA_SPEC_BONUS") or "4.0")
_TORQUE_INTENT_RE = re.compile(r"\btorque|\btightening|\bnm\b|\bnewton|\btighten", re.I)


def _torque_intent(query: str) -> bool:
    return bool(_TORQUE_INTENT_RE.search(query or ""))


def _rerank(rows, query: str):
    """Re-order bm25 candidate rows with the title-coverage + spec-table priors. Stable
    for ties (Python sort), so equal-prior docs keep their bm25 order."""
    if not rows:
        return rows
    qterms = [t for t in _terms(query) if len(t) >= 3]
    torque = _torque_intent(query)

    def adj(r):
        score = r["rank"]
        title = (r["title"] or "").lower()
        score -= _TITLE_TERM_BONUS * sum(1 for t in qterms if t in title)
        if torque and title.startswith("tightening torque"):
            score -= _SPEC_TABLE_BONUS
        return score

    return sorted(rows, key=adj)


# ── Cross-vehicle spec conflict flag (item 8) ────────────────────────────────
# The same component can carry DIFFERENT torque values across vehicles (the parked M6
# intake case: this car's spec table says 19 Nm in VCC-112365, an other-vehicle procedure
# says 20 Nm in VCC-131554). When a torque-intent result set contains BOTH a this-car
# value and a differing other-scope value for a component the query names, surface the
# disagreement so the model quotes THIS car's value and states scope — never silently
# averages or picks the wrong one.
_TORQUE_VAL_RE = re.compile(r"([A-Za-z][A-Za-z /()-]{2,40}?)\D{0,12}?(\d{1,3})\s*Nm", re.I)


def _torque_values_for(text: str, anchors: set[str]) -> set[int]:
    """Nm values whose immediately-preceding label text mentions a query component term."""
    out = set()
    for m in _TORQUE_VAL_RE.finditer(text or ""):
        label = m.group(1).lower()
        if any(a in label for a in anchors):
            out.add(int(m.group(2)))
    return out


def _torque_conflict(con, query: str, results: list[dict], car: CarContext | None = None) -> str | None:
    """If shown this-car and other-scope docs disagree on a component's torque, describe it."""
    car = car or CAR
    if not _torque_intent(query) or len(results) < 2:
        return None
    anchors = {t for t in _terms(query)
               if len(t) >= 3 and t not in _GENERIC_FIG and not _TORQUE_INTENT_RE.fullmatch(t)}
    if not anchors:
        return None
    this_vals, other_vals = set(), set()
    other_ref = None
    for r in results[:6]:
        scope = r.get("scope") or ""
        row = con.execute("SELECT text_md FROM documents WHERE doc_id = ?", (r["doc_id"],)).fetchone()
        vals = _torque_values_for(row[0] if row else "", anchors)
        if not vals:
            continue
        if scope == "this car" or scope.startswith("this car"):
            this_vals |= vals
        elif scope.startswith("other") or scope == "not vehicle-specific":
            if other_ref is None and (vals - this_vals):
                other_ref = r.get("vida_doc_ref")
            other_vals |= vals
    extra = sorted(other_vals - this_vals)
    if this_vals and extra:
        comp = " ".join(sorted(anchors))
        return (
            f"Conflicting torque values across vehicles for '{comp}': this car "
            f"({car.engine}) = {sorted(this_vals)} Nm, but other-vehicle docs list "
            f"{extra} Nm"
            + (f" (e.g. {other_ref})" if other_ref else "")
            + f". Use this car's value; do NOT quote another vehicle's torque."
        )
    return None


@_tool()
def search_docs(query: str, profile: str | None = None, limit: int = 8,
                car: str | None = None) -> list[dict]:
    """Full-text search over VIDA repair documents (procedures, specifications,
    installation instructions, parts info). Returns ranked results with short
    snippets and citations. Snippets are brief — follow up with get_document on
    promising hits (especially 'Tightening torque' and specification docs) to read
    full values before answering. VCC document references (e.g. 'VCC-112365-1')
    can be passed directly as the query OR straight to get_document.
    By DEFAULT results are tiered: documents VIDA marks applicable to {car}
    rank first with scope='this car', and the best matches from other vehicles
    are blended in with scope='other vehicles' — many P2 platform
    (S60/V70/XC70/S80/XC90) procedures are shared, so those are often equally
    valid. Pass profile='all' for a flat all-vehicle search, or a single short
    code ('B5244T5', 'XC90', '2004') to scope to another vehicle.
    Pass car='<engine code>' (e.g. car='B6294T', car='D5244T 2010') to retarget the
    WHOLE query — tiering AND the engine-applicability safety gate — to a DIFFERENT
    vehicle in this VIDA install, instead of the default {car}. Use it only when the user
    explicitly asks about another car; omit it for normal use."""
    # a per-query car override retargets scoping + the engine-family safety gate (item 7)
    active = _resolve_car(car) if car else CAR
    car_docs_sql = _build_car_docs_sql(active) if car else _CAR_DOCS_SQL
    tier1_sql = (f"{car_docs_sql} UNION SELECT doc_id FROM generic_docs") if car else _TIER1_SQL
    # a VCC doc reference resolves directly via metadata, not full-text
    vcc = re.search(r"VCC[- ]?(\d{6})(?:-\d+)?", query, re.I)
    with _con() as con:
        if vcc:
            rows = con.execute(
                "SELECT doc_id, title, profile, model, year, engine, section, db,"
                " table_name, source_pk, vida_doc_ref,"
                " substr(text_md, 1, 200) AS snip, 0 AS rank"
                " FROM documents WHERE vida_doc_ref LIKE ? LIMIT ?",
                (f"VCC-{vcc.group(1)}%", max(1, min(limit, 25))),
            ).fetchall()
            if rows:
                return _format_results(rows, con)
        # diagnostic-stub and parts-catalog docs only outrank procedures when the
        # question is actually about DTCs / part numbers — otherwise demote them
        qterms = set(_terms(query))
        diag_intent = qterms & {"dtc", "code", "codes", "fault", "diagnostic", "signal", "trouble"}
        parts_intent = bool(set(re.findall(r"[a-z]+", query.lower())) & {"part", "parts", "number", "oem", "pn"})
        penalty = (
            " + (CASE WHEN d.db = 'EPC' THEN ? ELSE 0 END)"
            " + (CASE WHEN d.title LIKE 'Diagnostic trouble code%'"
            "      OR d.title LIKE 'Signal too%' OR d.title LIKE 'Signal missing%'"
            "      OR d.title IN ('Faulty signal', 'Intermittent fault', 'Faulty software')"
            "      THEN ? ELSE 0 END)"
        )
        epc_pen = 0.0 if parts_intent else 6.0
        diag_pen = 0.0 if diag_intent else 5.0
        sql = (
            "SELECT d.doc_id, d.title, d.profile, d.model, d.year, d.engine, d.section,"
            " d.db, d.table_name, d.source_pk, d.vida_doc_ref,"
            # No highlight delimiters: '[',']' wrapped every matched token in literal
            # brackets, so a verbose match ("checking the oil level") came back as
            # "[Checking] [the] [oil] [level]" bracket-soup that models quoted verbatim into
            # answers. Plain snippet reads cleanly; full clean text is still in `lead`/get_document.
            " snippet(documents_fts, 1, '', '', ' … ', 24) AS snip,"
            f" bm25(documents_fts, 8.0, 1.0){penalty} AS rank"  # title weighted 8x
            " FROM documents_fts JOIN documents d ON d.rowid = documents_fts.rowid"
            " WHERE documents_fts MATCH ?"
        )
        lim = max(1, min(limit, 25))
        cand = max(lim, _CAND)  # fetch wider, then re-rank down to lim (item 2)
        profile_note = None
        yr = next((t for t in qterms if re.fullmatch(r"20(1[6-9]|[2-9]\d)", t)), None)
        if yr:
            profile_note = f"note: VIDA 2014D coverage ends around model year 2015 — {yr} content does not exist here"

        car_scope = bool(car) or profile is None or (profile or "").strip().lower() in {"", "this-car", "car", *CAR.aliases}

        def run(match: str, prof: str | None, car: bool = False):
            s, p = sql, [epc_pen, diag_pen, match]
            if car:
                # within tier 1, car-TAGGED docs outrank untagged generics at
                # comparable relevance (untagged short-title stubs otherwise crowd)
                s = s.replace(
                    " AS rank",
                    " + (CASE WHEN d.doc_id IN (SELECT doc_id FROM generic_docs) THEN 1.5 ELSE 0 END) AS rank",
                )
                s += f" AND d.doc_id IN ({tier1_sql})"
            elif prof:
                s += " AND (d.profile LIKE ? OR d.model LIKE ? OR d.year LIKE ? OR d.engine LIKE ?)"
                p += [f"%{prof}%"] * 4
            return con.execute(s + " ORDER BY rank LIMIT ?", p + [cand]).fetchall()

        scopes: dict[str, str] = {}
        if car_scope:
            # tiered, not filtered: car-tagged docs first, but the strongest
            # all-vehicle hits always blend in — P2-shared procedures are often
            # tagged only to a sibling model (V70/S80/XC90)
            relaxed_only = {"car": False, "all": False}

            def tier(car: bool, key: str):
                got = run(_fts_match(query), None, car=car)
                if len(got) < 2 and len(_terms(query)) >= 2:  # relax this tier independently
                    seen_t = {r["doc_id"] for r in got}
                    more = [r for r in run(_fts_match(query, relaxed=True), None, car=car) if r["doc_id"] not in seen_t]
                    relaxed_only[key] = not got and bool(more)
                    got = list(got) + more
                return _rerank(got, query)[:lim]  # item-2 re-rank, then trim to the result size

            car_rows, all_rows = tier(True, "car"), tier(False, "all")
            seen = {r["doc_id"] for r in car_rows}
            others = [r for r in all_rows if r["doc_id"] not in seen]
            # when the car tier holds only loose keyword matches but other
            # vehicles matched strictly, give the strong matches more room
            base_other = max(2, lim - len(car_rows))
            if relaxed_only["car"] and not relaxed_only["all"]:
                base_other = max(base_other, lim // 2)
            keep_other = min(len(others), base_other)
            rows = car_rows[: lim - keep_other] + others[:keep_other]
            generic = {
                r[0] for r in con.execute(
                    f"SELECT doc_id FROM generic_docs WHERE doc_id IN ({','.join('?' * len(rows))})",
                    [r["doc_id"] for r in rows],
                ).fetchall()
            } if rows else set()
            for r in car_rows:
                scopes[r["doc_id"]] = "not vehicle-specific" if r["doc_id"] in generic else "this car"
            for r in others:
                scopes[r["doc_id"]] = "other vehicles"
            if relaxed_only["car"] and relaxed_only["all"]:
                profile_note = ((profile_note + "; ") if profile_note else "") + \
                    "no strict matches — these are loose keyword matches only; the topic may not be covered"
        elif (profile or "").strip().lower() == "all":
            rows = run(_fts_quote(query), None)
        else:
            # free-text profile params ("2005 Volvo S60R B5254T4") match nothing as
            # a raw substring — fall back token-by-token, then drop the filter loudly
            rows = run(_fts_quote(query), profile)
            if not rows:
                for tok in sorted(re.findall(r"[A-Za-z0-9]+", profile), key=len, reverse=True):
                    rows = run(_fts_quote(query), tok)
                    if rows:
                        profile_note = f"profile filter '{profile}' matched nothing; used '{tok}' instead"
                        break
                else:
                    rows = run(_fts_quote(query), None)
                    profile_note = f"profile filter '{profile}' matched nothing; showing unfiltered results"
        if not car_scope and len(rows) < max(2, lim // 2):
            # strict all-terms match came up short — retry with OR semantics
            terms = re.findall(r"[A-Za-z0-9_.:/-]+", query)
            if len(terms) > 2:
                more = run(" OR ".join(f'"{t}"' for t in terms), profile)
                seen = {r["doc_id"] for r in rows}
                rows = list(rows) + [r for r in more if r["doc_id"] not in seen]
        if not car_scope:
            rows = _rerank(rows, query)[:lim]  # item-2 re-rank for explicit-vehicle searches too
        if car_scope and query and not vcc:
            # blend semantic recall on top of the lexical/tiered FTS result (item 1/3).
            # low_conf = the lexical side found NO strict match in either tier (all loose) —
            # only then may a confident vector hit outrank the loose lexical order.
            low_conf = bool(relaxed_only["car"] and relaxed_only["all"])
            rows, scopes = _augment_with_vectors(con, query, rows, scopes, lim, low_conf=low_conf,
                                                 car_docs_sql=(car_docs_sql if car else None))
        out = _format_results(rows, con, scopes, enforce_engine=car_scope,
                              car=(active if car else None))
        conflict = _torque_conflict(con, query, out, active)
        if conflict and out:
            out[0] = {**out[0], "conflict_warning": conflict}
        if profile_note and out:
            out[0] = {"notice": profile_note, **out[0]}
        return out


def _format_results(rows, con, scopes: dict[str, str] | None = None,
                    enforce_engine: bool = False, car: CarContext | None = None) -> list[dict]:
    """Shape ranked rows into result dicts. When enforce_engine is set (default car
    scope), engine applicability is ENFORCED, not just labelled: a doc for a different
    engine family keeps its slot but its quotable `lead` text is WITHHELD and it is
    flagged, so a single-pass model can never quote a wrong-engine torque/spec value.
    Left off for explicit cross-vehicle queries (profile='all'/<code>) and VCC lookups,
    where the caller has deliberately asked to see another vehicle's content. `car` (item
    7) makes the applicability gate and the warning text follow a per-query car."""
    active = car or CAR
    compat_re = re.compile(active.engine_family, re.I) if car else None
    out = []
    for i, r in enumerate(rows):
        appl = _engine_applicability(r["engine"], compat_re) if enforce_engine else "compatible"
        scope = scopes[r["doc_id"]] if scopes and r["doc_id"] in scopes else None
        if appl == "wrong":  # the soft "other vehicles" label understates the hazard
            scope = "other engine (not this car)"
        item = {
            "doc_id": r["doc_id"],
            "title": r["title"],
            "snippet": r["snip"],
            "vida_doc_ref": r["vida_doc_ref"],
            **({"scope": scope} if scope else {}),
            "model": _short(r["model"]),
            "year": _short(r["year"]),
            "engine": _short(r["engine"]),
            "section": r["section"],
            "citation": _doc_citation(r),
        }
        if appl == "wrong":
            item["applicability_warning"] = (
                f"This document is for a DIFFERENT engine ({_short(r['engine'], 40)}), "
                f"not this car's {active.engine}. Do NOT quote its torque, clearance, or "
                "specification values for this car."
            )
        if i < 3 and appl != "wrong":  # lead text lets single-pass clients answer spec questions
            lead = con.execute(
                "SELECT substr(text_md, 1, 2800) FROM documents WHERE doc_id = ?",
                (r["doc_id"],),
            ).fetchone()
            leadtxt = lead[0] if lead else None
            if enforce_engine and appl == "generic" and leadtxt and _lead_wrong_engine(leadtxt, compat_re):
                # Generic/untagged doc, but the START of its text is about a different
                # engine. The doc may still cover this car further down, so don't relabel
                # its scope — just withhold the wrong-engine lead and steer to the full text.
                item["applicability_warning"] = (
                    "The start of this document gives values for a DIFFERENT engine. "
                    "Lead withheld; do not quote it. Open it with get_document to find "
                    f"this car's {active.engine} section."
                )
            else:
                item["lead"] = leadtxt
        out.append(item)
    return out


@_tool()
def get_document(doc_id: str, offset: int = 0) -> dict:
    """Fetch the normalized text of one VIDA document by doc_id (or by VCC
    reference like 'VCC-112365-1'), with metadata, image reference ids, and its
    citation. Long documents are returned in 8000-character pages: if the result
    says it is truncated, call again with the given next_offset to continue."""
    PAGE = 8000
    with _con() as con:
        vcc = re.fullmatch(r"\s*(VCC[- ]?\d{6}(?:-\d+)?)\s*", doc_id, re.I)
        if vcc:
            ref = vcc.group(1).upper().replace(" ", "-")
            r = con.execute(
                "SELECT * FROM documents WHERE vida_doc_ref LIKE ? ORDER BY vida_doc_ref LIMIT 1",
                (f"{ref}%",),
            ).fetchone()
        else:
            r = con.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if r is None:
        return {"error": f"no document with doc_id {doc_id!r}"}
    profiles = json.loads(r["profile"] or "[]")
    full = r["text_md"] or ""
    page = full[offset : offset + PAGE]
    truncated = offset + PAGE < len(full)
    # An explicit fetch by id is not blocked, but if this doc is tagged to a different
    # engine family, warn so its torque/clearance values aren't quoted for this car.
    appl = _engine_applicability(r["engine"])
    out = {
        "doc_id": r["doc_id"],
        "title": r["title"],
        "vida_doc_ref": r["vida_doc_ref"],
        "profile_count": len(profiles),
        "profile_sample": profiles[:5],
        "model": _short(r["model"]),
        "year": _short(r["year"]),
        "engine": _short(r["engine"]),
        "ecu": r["ecu"],
        "section": r["section"],
        "text_md": page,
        "text_length": len(full),
        "truncated": truncated,
        **({"next_offset": offset + PAGE} if truncated else {}),
        "image_refs": json.loads(r["image_refs"] or "[]"),
        "citation": _doc_citation(r),
    }
    _durl = _doc_base_url()
    if _durl and r["vida_doc_ref"]:
        out["ref_url"] = f"{_durl}/doc/{r['vida_doc_ref']}"  # clickable source (Open WebUI)
    if appl == "wrong":
        out["applicability_warning"] = (
            f"This document is for a DIFFERENT engine ({_short(r['engine'], 40)}), not "
            f"this car's {CAR.engine}. Its torque, clearance, and specification values "
            "do not apply to this car — do not quote them."
        )
    return out


@_tool()
def lookup_pin(module: str | None = None, pin: str | None = None, query: str | None = None) -> list[dict]:
    """Look up wiring pinout rows for {car} (source: {pinout} wiring
    diagrams, community-verified CSV). Filter by module (e.g. 'ECM', 'CEM' —
    note the ABS/DSTC module is 'BCM', the amplifier 'AUM', airbags 'SRS'),
    pin (accepts '4', 'B4', or 'B:4' — connector letter optional), and/or a
    free-text query over function/notes/connected-component names. Results
    include wire color, where the wire goes, confidence, and a page-cited
    source; a trailing entry warns if results were truncated."""
    MODULE_ALIASES = {"ABS": "BCM", "DSTC": "BCM", "BRAKE": "BCM", "AIRBAG": "SRS",
                      "AMP": "AUM", "AMPLIFIER": "AUM", "ETM": "ECM"}
    LIMIT = 200
    where, params = [], []
    if module:
        m_up = module.strip().upper()
        where.append("upper(module) = ?")
        params.append(MODULE_ALIASES.get(m_up, m_up))
    if pin:
        m = re.fullmatch(r"\s*([A-Za-z])\s*:?\s*(\d+)\s*", pin)
        if m:
            where.append("upper(connector) = upper(?) AND pin = ?")
            params += [m.group(1), m.group(2)]
        else:
            where.append("pin = ?")
            params.append(pin.strip())
    if not where and not query:
        return [{"error": "provide at least one of module, pin, query"}]
    with _con() as con:
        def fetch(q_match: str | None):
            w, p = list(where), list(params)
            if q_match:
                w.append("row_id IN (SELECT rowid FROM pinouts_fts WHERE pinouts_fts MATCH ?)")
                p.append(q_match)
            return con.execute(
                f"SELECT * FROM pinouts WHERE {' AND '.join(w)}"
                f" ORDER BY module, connector, CAST(pin AS INTEGER) LIMIT {LIMIT + 1}",
                p,
            ).fetchall()

        rows = fetch(_fts_match(query)) if query else fetch(None)
        if query and not rows:  # strict AND came up empty — relax to any-term
            rows = fetch(_fts_match(query, relaxed=True))
        out = []
        for r in rows:
            goes_to_desc = None
            comp = con.execute(
                "SELECT description FROM components WHERE component_id = ?",
                (r["goes_to"],),
            ).fetchone()
            if comp:
                goes_to_desc = comp["description"]
            out.append(
                {
                    "module": r["module"],
                    "connector": r["connector"],
                    "pin": r["pin"],
                    "wire_color": r["wire_color"],
                    "goes_to": r["goes_to"],
                    "goes_to_description": goes_to_desc,
                    "source_pin": r["source_pin"],
                    "function": r["function"],
                    "confidence": r["confidence"],
                    "notes": r["notes"],
                    "citation": _pin_citation(r),
                    "source_row": f"{r['source_file']}:{r['csv_line']}",
                }
            )
    if len(out) > LIMIT:
        out = out[:LIMIT]
        out.append({"warning": f"results truncated at {LIMIT} rows — narrow by module/pin/query"})
    return out


def _table_count(con, table: str) -> int | None:
    """COUNT(*) for a table, or None if it doesn't exist (artifact not built yet)."""
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except sqlite3.OperationalError:
        return None


def _sidecar_count(db_path, table: str) -> int | None:
    """COUNT(*) in a read-only sidecar SQLite (images / vectors), or None if absent."""
    if not db_path.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
        con.execute("PRAGMA busy_timeout=1500")
        try:
            return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            con.close()
    except sqlite3.OperationalError:
        return None


def _artifact_status() -> dict:
    """Pre-flight (item 6): which build artifacts exist, with counts, so a consumer can
    see at a glance what ingestion step is still missing rather than discovering it as a
    silent capability gap mid-query. Each capability lists what to run if it's not ready."""
    with _con() as con:
        epc = _table_count(con, "epc_parts")
        syn = _table_count(con, "synonyms")
        gen = _table_count(con, "generic_docs")
    imgs = _sidecar_count(IMAGES_DB, "images")
    graphics = _sidecar_count(IMAGES_DB, "graphics")
    vecs = _sidecar_count(VECTORS_DB, "vec_map")
    embed_ok = _VECTORS_ENABLED and bool(_embed_query("ping"))
    live_sql = _sql("imagerepository")
    if live_sql is not None:
        live_sql.close()

    def cap(ready, detail, build=None):
        d = {"ready": bool(ready), **detail}
        if not ready and build:
            d["build"] = build
        return d

    return {
        "core_store": cap(True, {"path": "data/vida-kb.sqlite3",
                                 "epc_parts": epc, "generic_docs": gen, "synonyms": syn},
                          "uv run python src/extract_epc.py && uv run python src/build_synonyms.py"),
        "figures": cap(imgs, {"baked_images": imgs, "title_path_map": graphics},
                       "uv run python src/extract_images.py (needs vida-sql up once)"),
        "semantic_search": cap(vecs and embed_ok,
                               {"vectors": vecs, "embed_endpoint": EMBED_URL,
                                "embed_endpoint_reachable": embed_ok},
                               "uv run python src/build_embeddings.py (needs an embed endpoint up)"),
        "live_sql_fallback": cap(live_sql is not None,
                                 {"note": "optional — only used to serve figures/EPC not yet baked"}),
    }


@_tool()
def list_sources() -> dict:
    """Report what is indexed: VIDA document counts by database/section, pinout coverage
    by module, AND a pre-flight `artifacts` report — which build artifacts (EPC, baked
    figures, embeddings/vectors, synonyms) are present and whether the embedding endpoint
    and live SQL are reachable, so a consumer can see what ingestion step is still missing."""
    with _con() as con:
        docs_by_db = con.execute(
            "SELECT db, section, COUNT(*) n FROM documents GROUP BY db, section ORDER BY n DESC"
        ).fetchall()
        pins = con.execute(
            "SELECT module, COUNT(*) n FROM pinouts GROUP BY module ORDER BY module"
        ).fetchall()
        n_docs = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_comp = con.execute("SELECT COUNT(*) FROM components").fetchone()[0]
    return {
        "vehicle": CAR.label,
        "vida_documents": {
            "total": n_docs,
            "by_db_section": [dict(r) for r in docs_by_db],
        },
        "pinouts": {
            "vehicle": CAR.label,
            "source": CAR.pinout_source,
            "by_module": [dict(r) for r in pins],
            "components_indexed": n_comp,
        },
        "artifacts": _artifact_status(),
    }


# EPC part lookups read the LOCAL store (epc_parts + epc_section_profiles, built by
# extract_epc.py from the EPC SQL Server) — no live SQL at runtime. Applicability scope
# lives on the SECTION via ComponentConditions.fkProfile, materialized into
# epc_section_profiles; the one bound `?` in the CASE is this car's profile id.
_SCOPE_RANK = {"this car": 0, "not vehicle-specific": 1, "other vehicles": 2}


def _epc_scope_sql(alias: str = "ep") -> str:
    return (
        "CASE WHEN EXISTS (SELECT 1 FROM epc_section_profiles sp"
        f"                 WHERE sp.section_id = {alias}.section_id AND sp.profile_id = ?) THEN 'this car'"
        "     WHEN EXISTS (SELECT 1 FROM epc_section_profiles sp"
        f"                 WHERE sp.section_id = {alias}.section_id) THEN 'other vehicles'"
        "     ELSE 'not vehicle-specific' END"
    )


@_tool()
def epc_part(part_number: str | None = None, component: str | None = None,
             limit: int = 20) -> dict:
    """Exact part-number / component lookup in the Volvo EPC parts catalogue
    (read-only, from the local store). Use this — NOT search_docs — when the user
    gives or asks for a specific Volvo part number, or wants the part numbers
    for a named component (e.g. "oxygen sensor", "thermostat", "ignition coil").

    Pass exactly one of:
      part_number  a Volvo part number (e.g. "1271998"); returns its
                   description, the catalogue sections it appears in, and
                   whether each section applies to this car.
      component    a component name; returns matching part numbers, this-car
                   applicable ones first.

    Applicability scope per row: 'this car' (EPC marks the section applicable to
    {car}), 'other vehicles', or 'not vehicle-specific'. The
    same part can appear under several sections with different scopes. Each row
    cites EPC/PartItems/<part_number>."""
    if bool(part_number) == bool(component):
        return {"error": "pass exactly one of part_number or component"}
    lim = max(1, min(int(limit), 50))
    scope_case = _epc_scope_sql("ep")
    cols = (f"ep.part_number, ep.description, ep.function_group, ep.section_title,"
            f" {scope_case} AS scope")
    try:
        with _con() as con:
            if part_number:
                # models sometimes pass the number as a JSON int / quoted string
                pn = re.sub(r"\s+", "", str(part_number)).strip('"\'')
                raw = con.execute(
                    f"SELECT {cols} FROM epc_parts ep WHERE ep.part_number = ? LIMIT ?",
                    (CAR.profile_id, pn, lim),
                ).fetchall()
            else:
                # fetch extra raw rows so Python-side dedup can still fill `limit`
                raw = con.execute(
                    f"SELECT {cols} FROM epc_parts ep WHERE ep.description LIKE ? LIMIT ?",
                    (CAR.profile_id, f"%{str(component).strip()}%", lim * 6),
                ).fetchall()
    except sqlite3.OperationalError:
        return {"error": "EPC parts are not built into the local store yet — run "
                         "`uv run python src/extract_epc.py` (needs vida-sql up once).",
                "rows": []}
    raw = [
        {"part_number": r[0], "description": r[1], "function_group": r[2],
         "section": r[3], "scope": r[4], "citation": f"EPC/PartItems/{r[0]}"}
        for r in raw
    ]

    if part_number:
        rows = sorted(raw, key=lambda x: _SCOPE_RANK.get(x["scope"], 3))
        note = None if rows else "no EPC part with that exact number"
    else:
        # collapse to distinct part numbers, keeping the best (lowest-rank) scope
        best: dict[str, dict] = {}
        for r in raw:
            cur_best = best.get(r["part_number"])
            if cur_best is None or _SCOPE_RANK.get(r["scope"], 3) < _SCOPE_RANK.get(cur_best["scope"], 3):
                best[r["part_number"]] = r
        rows = sorted(best.values(), key=lambda x: (_SCOPE_RANK.get(x["scope"], 3), x["part_number"]))[:lim]
        note = None if rows else (
            "No catalogue part is named exactly that. The EPC uses Volvo's own generic "
            "names (seals are filed as 'Sealing ring' / 'Seal', not 'cam seal'; gaskets "
            "as 'Gasket'), so retrying epc_part with synonyms will keep returning nothing "
            "— STOP calling epc_part for this. Either answer from search_docs (it indexes "
            "the 'Parts: …' lists), or make ONE more epc_part call using only the single "
            "parent noun (e.g. 'camshaft', 'seal')."
        )
    out = {"query": {"part_number": part_number, "component": component}, "rows": rows}
    if note:
        out["note"] = note
    return out


# image_refs store VIDA's own paths "<16hex-id>_<W>_<H>.<ext>"; the bytes live in
# the attached imagerepository DB keyed by that exact path. Raster formats render
# in any browser; CGM is WebCGM vector (wiring/diagnostic diagrams) and needs an
# offline jcgm→SVG pass that is not built yet — those degrade to a labeled note.
_IMG_FORMAT = {".gif": "gif", ".jpeg": "jpeg", ".jpg": "jpeg", ".png": "png",
               ".svg": "svg"}  # svg = a CGM rendered offline by extract_cgm.py (Milestone 2)

# Figure bytes are served from the BAKED image store (extract_images.py) so the runtime
# needs no live SQL Server. The live imagerepository remains a transparent fallback for
# any figure not yet baked (e.g. a partial extraction).
IMAGES_DB = ROOT / "data" / "vida-images.sqlite3"


def _images_con():
    """Read-only connection to the baked image store, or None if it isn't built."""
    if not IMAGES_DB.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{IMAGES_DB}?mode=ro", uri=True, timeout=2)
        con.execute("PRAGMA busy_timeout=2000")
        return con
    except sqlite3.OperationalError:
        return None


def _local_image_blob(path: str):
    """(bytes, content_type) for a path from the baked store, or (None, None)."""
    con = _images_con()
    if con is None:
        return None, None
    try:
        row = con.execute(
            "SELECT data, content_type FROM images WHERE path = ?", (path,)
        ).fetchone()
        return (bytes(row[0]), row[1]) if row and row[0] else (None, None)
    except sqlite3.OperationalError:
        return None, None
    finally:
        con.close()

# VIDA docs reference their figures INLINE as "[image A2101030]" markers, where the
# letter+number is LocalizedGraphics.title. The marker's POSITION lets us score a
# figure by how well the text AROUND it covers the query — so "cylinder head torque
# sequence" returns the figure whose caption is "Tightening sequence for cylinder
# head bolts", not just any figure in any cylinder-head document. Term weights are
# IDF-style so discriminating words (torque, sequence) outweigh ubiquitous ones
# (cylinder, head). Prefixes encode figure kind: 'M' = EPC parts-catalogue exploded
# views (part call-outs); others (A/H/J/S/D…) = service-procedure & spec
# illustrations — a user asking to "see" something usually wants the latter, so
# M-figures are nudged back.
_IMG_MARK = re.compile(r"\[image\s+([A-Za-z]+\d+)\]")
_FIG_WINDOW = 240  # chars of text on each side of a marker that "caption" the figure
_M_FACTOR = 0.6    # parts-catalogue figures score lower than procedure illustrations


def _resolve_figure_doc(con, doc_id: str):
    """Resolve a doc_id or VCC reference to one (doc_id, title, image_refs, text_md,
    scope) row, or None."""
    cols = f"doc_id, title, image_refs, text_md, {_fig_scope_case('')} AS scope"
    vcc = re.fullmatch(r"\s*(VCC[- ]?\d{6}(?:-\d+)?)\s*", doc_id, re.I)
    if vcc:
        ref = vcc.group(1).upper().replace(" ", "-")
        return con.execute(
            f"SELECT {cols} FROM documents"
            " WHERE vida_doc_ref LIKE ? ORDER BY vida_doc_ref LIMIT 1",
            (f"{ref}%",),
        ).fetchone()
    return con.execute(
        f"SELECT {cols} FROM documents WHERE doc_id = ?", (doc_id,),
    ).fetchone()


# Documents whose figures may be shown for THIS car: VIDA-tagged to the car
# (exact profile + ancestors), engine-tagged to the exact engine on any body, or
# untagged/generic (shared boilerplate). Everything else — docs tagged to a
# different engine — is EXCLUDED from figure candidacy entirely, so a wrong-engine
# figure can never be ranked, let alone win, on text similarity. (The figure-text
# engine gate in _figure_scores then handles untagged docs that name a wrong engine
# only in prose.)
_FIG_APPLICABLE_SQL = (
    f"{_CAR_DOCS_SQL}"
    " UNION SELECT doc_id FROM generic_docs"
    f" UNION SELECT doc_id FROM documents WHERE {_engine_token_sql('engine')}"
)
def _fig_scope_case(alias: str = "d") -> str:
    """SQL CASE labelling a document's applicability scope for this car."""
    p = f"{alias}." if alias else ""
    return (
        f"CASE WHEN {p}doc_id IN ({_CAR_DOCS_SQL}) OR {_engine_token_sql(p + 'engine')}"
        f"      THEN 'this car ({CAR.engine})'"
        f"     WHEN {p}doc_id IN (SELECT doc_id FROM generic_docs)"
        "      THEN 'not vehicle-specific (shared illustration)'"
        "     ELSE 'engine-family' END"
    )


def _figure_docs_for_query(con, query: str, n: int = 12):
    """Rank APPLICABLE documents that have figures for a free-text query. Candidates
    are hard-restricted to this car / exact-engine / generic docs — wrong-engine docs
    are never candidates. Returns rows (doc_id, title, image_refs, text_md, scope).
    Relaxes to any-term on a miss."""
    sql = (
        "SELECT d.doc_id, d.title, d.image_refs, d.text_md,"
        f" {_fig_scope_case('d')} AS scope,"
        " bm25(documents_fts, 8.0, 1.0) AS rank"
        " FROM documents_fts JOIN documents d ON d.rowid = documents_fts.rowid"
        " WHERE documents_fts MATCH ?"
        " AND d.image_refs IS NOT NULL AND d.image_refs != '[]' AND d.image_refs != ''"
        f" AND d.doc_id IN ({_FIG_APPLICABLE_SQL})"
        " ORDER BY rank LIMIT ?"
    )
    rows = con.execute(sql, (_fts_match(query), n)).fetchall()
    if not rows and len(_terms(query)) >= 2:
        rows = con.execute(sql, (_fts_match(query, relaxed=True), n)).fetchall()
    return rows


def _query_terms(query: str) -> list[str]:
    """Distinct content terms (≥3 chars) used for figure scoring."""
    return [t for t in dict.fromkeys(_terms(query)) if len(t) >= 3]


def _term_idf(con, terms: list[str]) -> dict[str, float]:
    """IDF-style weight per term: rarer terms (e.g. 'sequence') weigh more than
    ubiquitous ones (e.g. 'cylinder'), so they dominate figure relevance."""
    n = con.execute("SELECT COUNT(*) FROM documents").fetchone()[0] or 1
    weights = {}
    for t in terms:
        try:
            df = con.execute(
                "SELECT COUNT(*) FROM documents_fts WHERE documents_fts MATCH ?",
                (f'"{t}"',),
            ).fetchone()[0]
        except sqlite3.OperationalError:
            df = 0
        weights[t] = math.log(1.0 + n / (df + 1))
    return weights


def _markers(text: str) -> list[tuple[int, str]]:
    """(position, figure-title) for each inline [image X#####] marker, in text order."""
    return [(m.start(), m.group(1)) for m in _IMG_MARK.finditer(text or "")]


def _figure_scores(text: str, query: str | None, idf: dict[str, float]):
    """For one doc, score each figure by how well the text around its marker covers
    the query terms (IDF-weighted), with parts-catalogue figures nudged back. A figure
    whose surrounding text names a DIFFERENT engine gets score -1 (ineligible) — this
    is the content-aware engine gate that catches untagged docs naming a wrong engine
    in prose. Returns (markers, scores, best_eligible_score); best is -1 when every
    figure is ineligible, 0 when there is nothing to rank on."""
    markers = _markers(text)
    if not markers:
        return markers, [], 0.0
    terms = list(idf)
    scores = []
    for pos, anum in markers:
        if _wrong_engine_near(text, pos):  # figure illustrates a different engine — exclude
            scores.append(-1.0)
            continue
        window = (text or "")[max(0, pos - _FIG_WINDOW): pos + _FIG_WINDOW].lower()
        s = sum(idf[t] for t in terms if t in window) if query else 0.0
        if anum[:1].upper() == "M":
            s *= _M_FACTOR
        scores.append(s)
    eligible = [s for s in scores if s >= 0]
    return markers, scores, (max(eligible) if eligible else -1.0)


# Words that describe a fastening/spec/display action rather than identify a
# component. Stripped before the relevance gate so "valve COVER torque sequence"
# must find the cover, not just match the generic "torque/bolt/sequence" vocabulary
# that appears in every tightening document.
_GENERIC_FIG = {
    "bolt", "bolts", "screw", "screws", "nut", "nuts", "torque", "torques",
    "tightening", "tighten", "sequence", "order", "stage", "stages", "spec",
    "specs", "specification", "value", "values", "setting", "settings", "diagram",
    "diagrams", "picture", "pictures", "image", "images", "show", "photo",
    "location", "position", "nm", "ftlb", "routing", "route",
    # NOTE: "pattern" is deliberately NOT stripped — it stays a required component term
    # so the gate keeps an anchor, but _SYN maps it to "sequence"/"order" so _covers()
    # still matches VIDA's wording. Stripping it left bare "head …" with no real anchor
    # and turned a safe refusal into a wrong figure (a seat-belt photo).
}
_GATE_WINDOW = 1200  # chars around a figure within which its component must appear


def _component_terms(query: str) -> list[str]:
    """Query terms that identify the component (generic torque/spec words removed)."""
    return [t for t in _query_terms(query) if t not in _GENERIC_FIG]


def _covers(term: str, window: str) -> bool:
    """True if the term, its singular stem, or a known synonym appears in the text."""
    cands = {term, term[:-1] if term.endswith("s") and len(term) > 3 else term}
    for s in _SYN.get(term, []):
        cands.add(s)
        cands.add(s[:-1] if s.endswith("s") and len(s) > 3 else s)
    return any(c in window for c in cands)


def _gate_missing(text: str, pos: int, query: str) -> list[str]:
    """Component terms NOT found near the figure marker. Empty list => the figure is
    on-topic; a non-empty list means showing it would be misleading (e.g. no valve-
    cover figure exists, so the closest match is an unrelated cylinder-head bolt)."""
    comp = _component_terms(query)
    if not comp:
        return []
    window = (text or "").lower()[max(0, pos - _GATE_WINDOW): pos + _GATE_WINDOW]
    return [t for t in comp if not _covers(t, window)]


def _ordered_paths(cur, img_con, refs: list[str], markers, scores) -> list[str]:
    """Order a doc's figure paths by relevance, EXCLUDING any figure whose marker was
    ruled ineligible (score < 0, e.g. wrong-engine). Eligible figures whose text best
    covers the query come first; remaining un-flagged figures follow in document order.
    Returns [] when every figure is ineligible. Falls back to document order when there
    is nothing to rank on (no markers). Marker title->path resolution (non-1:1 docs) uses
    the baked `graphics` map first, then the live LocalizedGraphics cursor."""
    if not markers or not scores:
        return refs
    eligible = [k for k in range(len(markers)) if scores[k] >= 0]
    if not eligible:
        return []  # every figure illustrates a different engine — show none

    def resolve(title, refset):  # marker title -> the path(s) among this doc's refs
        if img_con is not None:
            try:
                hits = {r[0] for r in img_con.execute(
                    "SELECT path FROM graphics WHERE title = ?", (title,))}
                hits &= refset
                if hits:
                    return hits
            except sqlite3.OperationalError:
                pass
        if cur is not None:
            cur.execute("SELECT path FROM LocalizedGraphics WHERE title = %s", (title,))
            return {r[0] for r in cur.fetchall()} & refset
        return set()

    order = sorted(eligible, key=lambda k: scores[k], reverse=True)
    one_to_one = len(markers) == len(refs)
    can_resolve = img_con is not None or cur is not None
    paths, excluded = [], set()
    if one_to_one or not can_resolve:  # 1:1 by text order (the common case) — zip directly
        paths = [refs[k] for k in order if k < len(refs)]
        excluded = {refs[k] for k in range(len(markers)) if k < len(refs) and scores[k] < 0}
    else:  # counts differ — resolve each marker's title -> path
        refset = set(refs)
        for k in order:
            hit = next(iter(resolve(markers[k][1], refset)), None)
            if hit:
                paths.append(hit)
        for k in range(len(markers)):  # collect wrong-engine paths so we don't re-add them
            if scores[k] < 0:
                excluded.update(resolve(markers[k][1], refset))
    seen, out = set(), []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    for p in refs:  # un-flagged figures with no marker we could rank — keep, but never the excluded ones
        if p and p not in seen and p not in excluded:
            seen.add(p)
            out.append(p)
    return out


def _fetch_figures(refs: list[str], lim: int, base_url, img_con, live_cur):
    """Resolve a de-duped, relevance-ordered ref list to displayable figures.
    Bytes come from the baked image store (img_con) first, with the live imagerepository
    cursor (live_cur) as a fallback. With base_url set, figures are served over HTTP and
    `shown` carries a body-render markdown URL (no bytes leave here); without it, bytes
    are returned as MCP Image blocks. Returns (images, shown, skipped)."""
    images, shown, skipped = [], [], []

    def _local(sql, path):  # baked-store read, resilient to a concurrent writer's lock
        if img_con is None:
            return None
        try:
            return img_con.execute(sql, (path,)).fetchone()
        except sqlite3.OperationalError:
            return None  # locked/absent — fall through to the live store

    def _blob(path):
        row = _local("SELECT data FROM images WHERE path = ?", path)
        if row and row[0]:
            return bytes(row[0])
        if live_cur is not None:
            live_cur.execute("SELECT imageData FROM LocalizedGraphics WHERE path = %s", (path,))
            row = live_cur.fetchone()
            if row and row[0]:
                return bytes(row[0])
        return None

    def _exists(path):
        if _local("SELECT 1 FROM images WHERE path = ?", path):
            return True
        if live_cur is not None:
            live_cur.execute("SELECT 1 FROM LocalizedGraphics WHERE path = %s", (path,))
            if live_cur.fetchone():
                return True
        return False

    for path in refs[:lim]:
        ext = os.path.splitext(path)[1].lower()
        serve_path = path
        if ext == ".cgm":
            # CGM Milestone 2: a vector diagram renders ONLY if extract_cgm.py baked a
            # rendered sibling (<stem>.png from jcgm-core, or .svg from a vector converter)
            # into the image store; otherwise it degrades to a labelled note.
            stem = re.sub(r"\.cgm$", "", path, flags=re.I)
            sib = next((stem + e for e in (".svg", ".png", ".gif", ".jpeg") if _exists(stem + e)), None)
            if sib:
                serve_path, ext = sib, os.path.splitext(sib)[1].lower()
            else:
                skipped.append({"id": path, "reason": "vector diagram (CGM) — not rendered yet"})
                continue
        fmt = _IMG_FORMAT.get(ext)
        if fmt is None:
            skipped.append({"id": path, "reason": f"unsupported format {ext}"})
            continue
        n = len(shown) + 1
        if base_url:
            if not _exists(serve_path):
                skipped.append({"id": path, "reason": "not found in image store"})
                continue
            url = f"{base_url}/img/{serve_path}"
            shown.append({"n": n, "id": path, "url": url,
                          "markdown": f"![Figure {n}]({url})"})
        else:
            data = _blob(serve_path)
            if not data:
                skipped.append({"id": path, "reason": "not found in image store"})
                continue
            images.append(Image(data=data, format=fmt))
            shown.append({"n": n, "id": path})
    return images, shown, skipped


@_tool()
def get_figures(query: str | None = None, doc_id: str | None = None, limit: int = 8):
    """Display VIDA figures/illustrations (wiring layouts, exploded parts views,
    location photos, torque-sequence diagrams) directly to the USER, inline in the
    chat. The images render for the user — you do NOT receive the pixels and cannot
    read them, so describe a figure only from the document's text, never invent it.

    THE usual way to use this is in ONE call with `query`: pass the user's topic
    (e.g. get_figures(query="cylinder head bolt torque sequence") or
    query="ECM connector location"). This finds the document whose matching text
    sits closest to a figure and shows that figure first — you do NOT need to
    search_docs or get_document first. Alternatively pass `doc_id` (or a VCC
    reference like 'VCC-112365-1') to show a specific document's figures, optionally
    with `query` to pick the most relevant ones within it. `limit` caps how many are
    shown (default 8); figures come back most-relevant first, so a small limit still
    returns the right one.

    Call this whenever the user asks to see/show a figure, diagram, illustration,
    picture, or image. Do NOT conclude "there are no figures" from a text search —
    only this tool knows; call it before saying images are unavailable.

    Returns a short text summary for you (the matched document + which figures were
    shown/skipped) plus the images, which the interface attaches to the reply.
    Reference figures by their number; do not attempt to embed image data."""
    if not query and not doc_id:
        return {"error": "pass query (a topic) or doc_id"}
    lim = max(1, min(int(limit), 12))
    base_url = _image_base_url()  # set => figures render in the answer body via URL

    # Build the ordered list of candidate docs to pull figures from, plus the
    # per-term IDF weights used to score figure relevance.
    with _con() as con:
        idf = _term_idf(con, _query_terms(query)) if query else {}
        if doc_id:
            r = _resolve_figure_doc(con, doc_id)
            if r is None:
                return {"error": f"no document with doc_id {doc_id!r}"}
            candidates = [r]
        else:
            candidates = _figure_docs_for_query(con, query)
            if not candidates:
                return {"query": query,
                        "note": "No document with figures matched that query. Try "
                                "different wording (component or system name), or open "
                                "a document with get_document and pass its doc_id."}

    img_con = _images_con()            # baked, self-contained store (preferred)
    con_img = _sql("imagerepository")  # live SQL — optional fallback / title resolution
    if img_con is None and con_img is None:
        return {"error": "no image store available — bake figures with "
                         "`uv run python src/extract_images.py`, or start vida-sql."}

    # Score each candidate by how well the text around its best figure covers the
    # query (wrong-engine figures already scored -1 by _figure_scores), then prefer the
    # doc with the most on-topic figure. Within the chosen doc, figures are ordered
    # most-relevant first and engine-ineligible ones dropped. doc_id mode keeps the doc.
    near_miss = None
    engine_blocked = False
    chosen = None
    try:
        cur = con_img.cursor() if con_img is not None else None
        scored = []
        for r in candidates:
            refs = list(dict.fromkeys(json.loads(r["image_refs"] or "[]")))
            markers, scores, best = _figure_scores(r["text_md"], query, idf)
            scored.append((best, r, refs, markers, scores))
        if query and not doc_id:
            scored.sort(key=lambda s: s[0], reverse=True)  # best figure coverage first

        for best, r, refs, markers, scores in scored:
            if not refs:
                if doc_id:  # the specific doc was asked for — report it has none
                    chosen = (r, refs, [], [], [])
                    break
                continue
            ordered = _ordered_paths(cur, img_con, refs, markers, scores)
            if query and not doc_id and not ordered:
                # this applicable doc's figures all illustrate a DIFFERENT engine
                engine_blocked = True
                continue
            # Relevance gate (query mode): judge the best ELIGIBLE figure. If its
            # surrounding text doesn't name the component asked about, refuse outright —
            # do NOT fall through to lower-ranked figures that merely share vocabulary.
            if query and not doc_id and markers and scores:
                eligible = [k for k in range(len(markers)) if scores[k] >= 0]
                if eligible:
                    top = max(eligible, key=lambda i: scores[i])
                    missing = _gate_missing(r["text_md"], markers[top][0], query)
                    if missing:
                        near_miss = (r["title"], missing)
                        break
            images, shown, skipped = _fetch_figures(ordered, lim, base_url, img_con, cur)
            if shown or doc_id:  # query mode needs ≥1 displayable figure (URL or bytes)
                chosen = (r, ordered, images, shown, skipped)
                break
    finally:
        if con_img is not None:
            con_img.close()
        if img_con is not None:
            img_con.close()

    if chosen is None:
        if near_miss:  # candidates existed but none actually depicted the component
            title, missing = near_miss
            return {"query": query, "closest_document": title,
                    "note": f"No VIDA figure clearly matches '{query}'. The closest "
                            f"document ('{title}') does not illustrate "
                            f"{', '.join(missing)}, so no figure is shown rather than a "
                            f"misleading one. VIDA likely has no dedicated diagram here "
                            f"— answer from search_docs (e.g. give the torque value) and "
                            f"tell the user there is no specific figure for this."}
        if engine_blocked:  # the only matching figures were for another engine
            return {"query": query,
                    "note": f"The figures matching '{query}' belong to a DIFFERENT "
                            f"engine, not this car's {CAR.engine}. VIDA has no figure "
                            f"for this on the {CAR.engine}. Do not show another "
                            f"engine's diagram — answer from search_docs "
                            f"(give the torque/spec values from text) and tell the user "
                            f"there is no engine-specific figure."}
        return {"query": query,
                "note": "Matched documents but none had displayable figures (their "
                        "graphics may be CGM vector diagrams, not yet rendered)."}

    r, refs, images, shown, skipped = chosen
    served = "served via URL — embed the markdown below" if base_url else "displayed above"
    summary = {
        "doc_id": r["doc_id"],
        "title": r["title"],
        "scope": r["scope"],
        "figures_shown_to_user": len(shown),
        "figures": shown,
        "note": (f"{len(shown)} figure(s) {served}. Refer to them by number; you cannot "
                 "see their contents." if shown
                 else "This document references no figures." if not refs
                 else "This document's figures could not be displayed (likely CGM "
                      "vector diagrams, not yet rendered)."),
    }
    if query:
        summary["matched_from_query"] = query
    if skipped:
        summary["skipped"] = skipped
    if len(refs) > lim:
        summary["truncated"] = (f"showing {lim} of {len(refs)} figures — "
                                f"call again with limit={min(len(refs), 12)} for more")
    if base_url and shown:
        # Figures are served over HTTP — the interface does NOT attach them. The model
        # MUST paste this markdown verbatim into its reply for the user to see them.
        summary["render_in_answer"] = "\n".join(s["markdown"] for s in shown)
        summary["note"] += (" IMPORTANT: include the markdown in 'render_in_answer' "
                            "VERBATIM in your reply so the figure(s) appear inline.")
        return summary
    return [summary, *images]


# ── Optional HTTP image server ───────────────────────────────────────────────
# get_figures returns MCP Image blocks by default, which Open WebUI renders inside
# the collapsed tool-call section. When this little server is up, get_figures instead
# returns markdown image URLs so the figure renders in the answer BODY. It binds ONLY
# to the private interface (VIDA_IMAGE_HOST), so it is reachable by the iPhone PWA
# over that network but is NOT exposed on the wider LAN. If the bind fails (that
# interface down), get_figures transparently falls back to MCP Image blocks.
IMAGE_HOST = _env("VIDA_IMAGE_HOST") or "localhost"
IMAGE_PORT = int(_env("VIDA_IMAGE_PORT") or "8766")
_IMAGE_PATH_RE = re.compile(r"[0-9a-fA-F]+_\d+_\d+\.(?:gif|jpe?g|png|svg)")
_IMAGE_SERVER_UP = False


def _image_base_url() -> str | None:
    return f"http://{IMAGE_HOST}:{IMAGE_PORT}" if _IMAGE_SERVER_UP else None


def _doc_base_url() -> str | None:
    """Base URL for the /doc/<VCC-ref> route (same private-network-bound server as /img/). When
    up, a cited token like 'VCC-461742-2' resolves to a readable HTML page of the source
    document — so citations become CLICKABLE in the Open WebUI answer body instead of inert
    text. Pair with the docs/openwebui/ outlet filter that rewrites VCC tokens to links."""
    return f"http://{IMAGE_HOST}:{IMAGE_PORT}" if _IMAGE_SERVER_UP else None


_DOC_REF_RE = re.compile(r"VCC-?\d{6}(?:-\d+)?", re.I)


def _doc_content(ref: str):
    """(html_bytes, content-type) for a VIDA document by VCC reference, or (None, None).
    Read-only; the ref is validated against a strict pattern and the lookup is parameterized
    (same resolver as get_document). text_md is HTML-escaped and shown in a <pre> block, so
    there is no injection surface and the formatting is preserved verbatim."""
    import html as _html
    if not ref or not _DOC_REF_RE.fullmatch(ref):
        return None, None
    refn = ref.upper().replace(" ", "-")
    with _con() as con:
        r = con.execute(
            "SELECT title, vida_doc_ref, model, year, engine, text_md FROM documents"
            " WHERE vida_doc_ref LIKE ? ORDER BY vida_doc_ref LIMIT 1", (f"{refn}%",)
        ).fetchone()
    if r is None:
        return None, None
    title = _html.escape(r["title"] or refn)
    scope = _html.escape(" · ".join(x for x in (r["model"], r["year"], r["engine"]) if x))
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title} — {_html.escape(r['vida_doc_ref'] or refn)}</title>"
        "<style>body{font:16px/1.55 -apple-system,system-ui,sans-serif;max-width:760px;"
        "margin:2rem auto;padding:0 1rem;color:#1a1a1a}h1{font-size:1.3rem;margin:0 0 .2rem}"
        ".ref{color:#888;font-size:.85rem;margin-bottom:1rem}pre{white-space:pre-wrap;"
        "word-wrap:break-word;font:inherit}hr{border:0;border-top:1px solid #ddd}</style></head>"
        f"<body><h1>{title}</h1><div class='ref'>{_html.escape(r['vida_doc_ref'] or refn)}"
        f"{(' — ' + scope) if scope else ''}</div><hr><pre>{_html.escape(r['text_md'] or '')}"
        "</pre></body></html>"
    )
    return page.encode("utf-8"), "text/html; charset=utf-8"


def _image_content(path: str):
    """(bytes, content-type) for a VIDA image path, or (None, None). Path is validated
    against a strict pattern, and the lookup is parameterized — no injection surface.
    Serves from the baked image store first (no live SQL), live imagerepository as
    fallback. A fresh SQLite connection per call keeps it safe across server threads."""
    if not _IMAGE_PATH_RE.fullmatch(path):
        return None, None
    ctype = {".gif": "image/gif", ".jpeg": "image/jpeg", ".jpg": "image/jpeg",
             ".png": "image/png", ".svg": "image/svg+xml"}.get(os.path.splitext(path)[1].lower())
    if ctype is None:
        return None, None
    data, lctype = _local_image_blob(path)   # baked store first
    if data:
        return data, lctype or ctype
    con = _sql("imagerepository")             # live fallback
    if con is None:
        return None, None
    try:
        cur = con.cursor()
        cur.execute("SELECT imageData FROM LocalizedGraphics WHERE path = %s", (path,))
        row = cur.fetchone()
        return (bytes(row[0]), ctype) if row and row[0] else (None, None)
    finally:
        con.close()


def _start_image_server() -> None:
    """Start the private-network-bound figure server in a daemon thread (best-effort)."""
    global _IMAGE_SERVER_UP
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # silence per-request logging
            pass

        def do_GET(self):
            path = self.path or ""
            m_img = re.fullmatch(r"/img/([^/?#]+)", path)
            m_doc = re.fullmatch(r"/doc/([^/?#]+)", path)
            if m_img:
                data, ctype = _image_content(m_img.group(1))
            elif m_doc:
                data, ctype = _doc_content(m_doc.group(1))   # cited VCC ref -> readable page
            else:
                data, ctype = None, None
            if data is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            self.wfile.write(data)

    try:
        srv = ThreadingHTTPServer((IMAGE_HOST, IMAGE_PORT), _Handler)
    except OSError as e:
        print(f"image server NOT started ({IMAGE_HOST}:{IMAGE_PORT}: {e}); "
              "figures will use the collapsed tool view", file=sys.stderr)
        return
    _IMAGE_SERVER_UP = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"figure image server on http://{IMAGE_HOST}:{IMAGE_PORT}", file=sys.stderr)


if __name__ == "__main__":
    if "--http" in sys.argv:
        idx = sys.argv.index("--http")
        port = int(sys.argv[idx + 1]) if len(sys.argv) > idx + 1 else 8765
        _start_image_server()
        mcp.settings.host = "127.0.0.1"
        mcp.settings.port = port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
