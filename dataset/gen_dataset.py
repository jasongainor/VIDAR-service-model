"""dataset/gen_dataset.py — build the function-calling training JSONL (DECISIONS D1).

Reads data/vida-kb.sqlite3 (Mac-side, stdlib only) and emits one gc.Record per line
in the canonical contract shape, mixed per gc.EXAMPLE_TYPES. Idempotent + deterministic
(ordered sources + --seed). Excludes any split_key reserved by the frozen benchmark
(vida-eval/frozen_split.json) so the eval set is never trained on (D16). Reports exact
per-type counts and any shortfall — no silent caps.

  python -m dataset.gen_dataset --limit 5000 --out data/ft/train.jsonl
  python dataset/gen_dataset.py --limit 200 --out data/ft/sample.jsonl   # quick sample

Only the produced JSONL crosses to the coordinator later; the 4 GB DBs stay on the Mac.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
from typing import Callable, Iterator, Optional

from grounding import contract as gc

from . import examples as ex
from . import sources as src


def wire_live_retriever() -> None:
    """Point the example builders at the live MCP retriever so every grounded
    tool-result is byte-identical to what the model meets at serve time (the realism
    fix — train == serve). Imports src/server.py lazily (it needs the embeddings
    backend); raises if unavailable so a real run never silently ships synthetic,
    skewed (single clean hit) data. Mac-side only — that's where the corpus +
    embeddings live; the box trains on the shipped JSONL (docs/DECISIONS.md)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(here, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import server  # noqa: E402  the MCP tool implementations (search_docs + embeddings)
    ex.set_retriever(search=lambda q, car=None: server.search_docs(q, car=car),
                     get_document=lambda doc_id: server.get_document(doc_id))

# How the grounded budget (gc.EXAMPLE_TYPES['grounded'].share) splits across the four
# grounded sub-kinds. Pinout is a small specialist slice (one car, 433 rows).
_GROUNDED_MIX = {"part": 0.40, "torque": 0.30, "procedure": 0.25, "pinout": 0.05}

# A record longer than the training window has its final answer (which is LAST) truncated
# away during SFT — i.e. the model trains on a headless example, the opposite of what we
# want. With realistic multi-hit tool results some procedure/torque examples are large,
# so we DROP any that wouldn't fit (and REPORT the count — no silent caps). Token estimate
# is conservative (chars/3.5 over-counts vs the real ~chars/4, so we never keep a too-long
# one). Set from --max-seq-length to match the training config.
_MAX_TOKENS = 8192
_dropped_overlong = 0


def _est_tokens(rec: gc.Record) -> int:
    chars = sum(len(m.get("content") or "")
                + len(json.dumps(m.get("tool_calls") or "", ensure_ascii=False))
                for m in rec.messages)
    return int(chars / 3.5)


# --------------------------------------------------------------------------- #
# Producers — each yields contract Records lazily from real rows.
# --------------------------------------------------------------------------- #
def _g_torque(conn: sqlite3.Connection) -> Iterator[gc.Record]:
    for d in src.iter_torque_docs(conn):
        r = ex.build_grounded_torque(d)
        if r:
            yield r


def _g_part(conn: sqlite3.Connection) -> Iterator[gc.Record]:
    for p in src.iter_parts(conn):
        r = ex.build_grounded_part(conn, p)
        if r:
            yield r


def _g_procedure(conn: sqlite3.Connection) -> Iterator[gc.Record]:
    for d in src.iter_procedure_docs(conn):
        r = ex.build_grounded_procedure(d)
        if r:
            yield r


def _g_pinout(conn: sqlite3.Connection) -> Iterator[gc.Record]:
    for row in src.iter_pinouts(conn):
        r = ex.build_grounded_pinout(row)
        if r:
            yield r


def _g_refusal(conn: sqlite3.Connection) -> Iterator[gc.Record]:
    for d in src.iter_procedure_docs(conn):
        r = ex.build_refusal(d)
        if r:
            yield r


def _g_elicitation(conn: sqlite3.Connection) -> Iterator[gc.Record]:
    for d in src.iter_procedure_docs(conn):
        r = ex.build_elicitation(d)
        if r:
            yield r


def _g_cross_vehicle(conn: sqlite3.Connection) -> Iterator[gc.Record]:
    for p in src.iter_parts(conn):
        r = ex.build_cross_vehicle(conn, p)
        if r:
            yield r


def _g_figure(conn: sqlite3.Connection) -> Iterator[gc.Record]:
    for d in src.iter_figure_docs(conn):
        r = ex.build_figure(d)
        if r:
            yield r


def _g_multi_turn(conn: sqlite3.Connection) -> Iterator[gc.Record]:
    """Pair two different-topic docs for the SAME concrete model."""
    by_model: dict[str, list[src.Doc]] = {}
    for d in src.iter_procedure_docs(conn):
        veh = src.pick_vehicle_for_doc(d)
        if not veh.model:
            continue
        by_model.setdefault(veh.model, []).append(d)
        # yield as soon as a model has a usable pair, then reset its bucket
        if len(by_model[veh.model]) >= 2:
            a, b = by_model[veh.model][0], by_model[veh.model][1]
            by_model[veh.model] = []
            r = ex.build_multi_turn(a, b)
            if r:
                yield r


# example_type -> ordered list of producers contributing to it
_PRODUCERS: dict[str, list[Callable[[sqlite3.Connection], Iterator[gc.Record]]]] = {
    "grounded": [],  # filled below with sub-kind allocation handled in plan()
    "refusal": [_g_refusal],
    "elicitation": [_g_elicitation],
    "cross_vehicle": [_g_cross_vehicle],
    "multi_turn": [_g_multi_turn],
    "figure": [_g_figure],
}

# grounded sub-producers keyed by sub-kind (allocated by _GROUNDED_MIX)
_GROUNDED_PRODUCERS = {
    "part": _g_part, "torque": _g_torque,
    "procedure": _g_procedure, "pinout": _g_pinout,
}


# --------------------------------------------------------------------------- #
# Planning + generation
# --------------------------------------------------------------------------- #
def plan_counts(limit: int) -> dict[str, int]:
    """Per-example-type target counts from gc.EXAMPLE_TYPES shares."""
    return {t: max(1, round(limit * spec["share"])) for t, spec in gc.EXAMPLE_TYPES.items()}


def _rec_hash(rec: gc.Record) -> str:
    return hashlib.sha1(json.dumps(rec.messages, sort_keys=True).encode()).hexdigest()


def _fill(producer: Iterator[gc.Record], target: int, *, seen: set[str],
          frozen_keys: set[str], frozen_sources: set[str]) -> list[gc.Record]:
    """Pull up to `target` deduped, non-frozen, contract-valid records. A record is
    frozen out if its split_key is reserved (held-out vehicle×topic cell) OR any of
    its source_ids is in the frozen benchmark (exact no-leak backstop)."""
    out: list[gc.Record] = []
    for rec in producer:
        if len(out) >= target:
            break
        if rec.metadata.get("split_key") in frozen_keys:
            continue
        if frozen_sources and any(s in frozen_sources for s in rec.metadata.get("source_ids", [])):
            continue
        if _est_tokens(rec) > _MAX_TOKENS:  # would truncate the answer in training -> drop + count
            global _dropped_overlong
            _dropped_overlong += 1
            continue
        h = _rec_hash(rec)
        if h in seen:
            continue
        if gc.validate_messages(rec.messages):  # never emit an invalid record
            continue
        seen.add(h)
        out.append(rec)
    return out


def generate(conn: sqlite3.Connection, limit: int, *, frozen_keys: set[str],
             frozen_sources: set[str], rng: random.Random, max_tokens: int = _MAX_TOKENS
             ) -> tuple[list[gc.Record], dict[str, int]]:
    global _MAX_TOKENS, _dropped_overlong
    _MAX_TOKENS = max_tokens
    _dropped_overlong = 0
    targets = plan_counts(limit)
    seen: set[str] = set()
    records: list[gc.Record] = []
    produced: dict[str, int] = {}

    # grounded: split its budget across sub-kinds
    g_target = targets["grounded"]
    for sub, frac in _GROUNDED_MIX.items():
        sub_target = max(1, round(g_target * frac))
        recs = _fill(_GROUNDED_PRODUCERS[sub](conn), sub_target, seen=seen,
                     frozen_keys=frozen_keys, frozen_sources=frozen_sources)
        records.extend(recs)
        produced[f"grounded:{sub}"] = len(recs)

    for etype, producers in _PRODUCERS.items():
        if etype == "grounded":
            continue
        got: list[gc.Record] = []
        for prod in producers:
            got.extend(_fill(prod(conn), targets[etype] - len(got), seen=seen,
                             frozen_keys=frozen_keys, frozen_sources=frozen_sources))
            if len(got) >= targets[etype]:
                break
        records.extend(got)
        produced[etype] = len(got)

    rng.shuffle(records)  # mix types so a training shard isn't blocky
    return records, {"targets": targets, "produced": produced,
                     "dropped_overlong": _dropped_overlong, "max_tokens": _MAX_TOKENS}


def load_frozen(path: Optional[str]) -> tuple[set[str], set[str]]:
    """Return (split_keys, source_ids) the benchmark reserved. Accepts the
    benchmark_gen.py frozen_split.json ({"split_keys":[...], "source_ids":[...]}) or a
    bare list of split_keys. Empty sets when absent."""
    if not path or not os.path.exists(path):
        return set(), set()
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return set(data.get("split_keys") or []), set(data.get("source_ids") or [])
    return set(data or []), set()


def write_jsonl(records: list[gc.Record], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(rec.to_json() + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="generate VIDA function-calling training JSONL")
    ap.add_argument("--limit", type=int, default=2000, help="approx total record count")
    ap.add_argument("--out", default="data/ft/train.jsonl")
    ap.add_argument("--kb", default=None, help="path to vida-kb.sqlite3 (or $VIDA_KB_DB)")
    ap.add_argument("--frozen", default="vida-eval/frozen_split.json",
                    help="split_keys to EXCLUDE (the frozen benchmark)")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--max-seq-length", type=int, default=8192,
                    help="training window; records whose answer would truncate past it "
                         "are dropped + reported (MUST match the campaign max_seq_length)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use hand-distilled single-hit tool results instead of the "
                         "live retriever (no server/embeddings needed) — produces "
                         "train/serve SKEW; NOT for real runs, debugging only")
    args = ap.parse_args(argv)

    if args.synthetic:
        print("retrieval mode: SYNTHETIC single-hit (skewed — debugging only)")
    else:
        try:
            wire_live_retriever()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"live retriever unavailable ({e!r}). Start the embeddings backend and "
                f"ensure src/server.py imports, OR pass --synthetic to deliberately "
                f"generate skewed single-hit data. Refusing to silently ship skew.")
        print("retrieval mode: LIVE server-backed (train == serve)")

    conn = src.connect(args.kb)
    try:
        rng = random.Random(args.seed)
        frozen_keys, frozen_sources = load_frozen(args.frozen)
        records, report = generate(conn, args.limit, frozen_keys=frozen_keys,
                                   frozen_sources=frozen_sources, rng=rng,
                                   max_tokens=args.max_seq_length)
    finally:
        conn.close()

    write_jsonl(records, args.out)
    print(f"wrote {len(records)} records -> {args.out}")
    print(f"frozen excluded: {len(frozen_keys)} split_keys, {len(frozen_sources)} source_ids")
    print(f"dropped (would truncate at max_seq_length={report['max_tokens']}): {report['dropped_overlong']}")
    print("per-type produced (target):")
    targets = report["targets"]
    for k, v in report["produced"].items():
        base = k.split(":", 1)[0]
        tgt = targets.get(base, "?")
        short = "  <-- SHORT" if isinstance(tgt, int) and v < (tgt if ":" not in k else 1) else ""
        print(f"  {k:22} {v}{short}")
    return 0


if __name__ == "__main__":
    import sys as _sys

    # If no args AND the KB is present, run a tiny smoke generation + validate.
    if len(_sys.argv) == 1:
        p = src.db_path()
        if not os.path.exists(p):
            print(f"SKIP self-check: KB absent at {p}")
        else:
            conn = src.connect()
            try:
                recs, report = generate(conn, limit=40, frozen_keys=set(),
                                        frozen_sources=set(), rng=random.Random(0))
            finally:
                conn.close()
            assert recs, "no records generated"
            bad = [r for r in recs if gc.validate_messages(r.messages)]
            assert not bad, f"{len(bad)} invalid records"
            types = {}
            for r in recs:
                types[r.metadata["example_type"]] = types.get(r.metadata["example_type"], 0) + 1

            # No-leak: freezing a real split_key AND a real source_id must EXCLUDE any
            # record carrying either from a fresh generation (D16). The previous
            # self-check only ran with empty frozen sets, so it never proved exclusion.
            key = next((r.metadata["split_key"] for r in recs if r.metadata.get("split_key")), None)
            srcid = next((r.metadata["source_ids"][0] for r in recs if r.metadata.get("source_ids")), None)
            conn2 = src.connect()
            try:
                recs2, _ = generate(conn2, limit=40, frozen_keys={key} if key else set(),
                                    frozen_sources={srcid} if srcid else set(),
                                    rng=random.Random(0))
            finally:
                conn2.close()
            if key:
                assert all(r.metadata.get("split_key") != key for r in recs2), \
                    f"frozen split_key {key!r} leaked into training"
            if srcid:
                assert all(srcid not in (r.metadata.get("source_ids") or []) for r in recs2), \
                    f"frozen source_id {srcid!r} leaked into training"
            print(f"gen_dataset.py self-check OK: {len(recs)} valid records; types={types}; "
                  f"no-leak verified (excluded split_key={key!r}, source_id={srcid!r})")
    else:
        raise SystemExit(main())
