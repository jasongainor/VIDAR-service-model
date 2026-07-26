"""vida-eval/benchmark_gen.py — build the FROZEN known-answer benchmark (Part 10 / D16).

Stratified, stitched from the same DB as training but HELD OUT: every case's
split_key (and exact source ids) are written to vida-eval/frozen_split.json, which
dataset/gen_dataset.py excludes — so the benchmark is never trained on. Split is by
vehicle AND topic (split_key), never random (VIDA rows duplicate across platform-mates).

Each case is a gc-shaped record: a system+user PREFIX the candidate model replays
(it produces the tool calls + answer itself), plus the gold answer in metadata that
vida-eval/score.py grades against:
  metadata: {example_type, answer_type, split_key, vehicle,
             gold_value | gold_part | gold_doc_ref | gold_passage | expect_refusal}

  python vida-eval/benchmark_gen.py --out vida-eval/benchmark.jsonl --freeze

Stdlib only; reuses dataset/sources.py + dataset/torque.py + grounding/contract.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# vida-eval/ is a hyphen dir; ensure the repo root is importable for grounding/dataset.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from grounding import contract as gc          # noqa: E402
from dataset import sources as src             # noqa: E402
from dataset.torque import first_torque        # noqa: E402

# Default per-category counts (~570 cases) — Part 10 stratification.
DEFAULT_COUNTS = {
    "part_number": 150, "torque": 150, "procedure": 80,
    "pinout": 60, "figure": 50, "cross_vehicle": 40, "refusal": 40,
}


def _case(cat: str, idx: int, question: str, vehicle, gold: dict) -> dict:
    veh = vehicle if (vehicle and not vehicle.is_empty()) else None
    return {
        "id": f"{cat}-{idx}",
        "messages": [gc.system_msg(), gc.user_msg(question, vehicle=veh)],
        "metadata": {
            "example_type": gold.get("example_type", cat),
            "answer_type": cat,
            "split_key": gold.get("split_key", f"{(veh.model if veh else 'any')}|{cat}"),
            "vehicle": veh.to_meta() if veh else None,
            **{k: v for k, v in gold.items() if k not in ("example_type", "split_key")},
        },
    }


# --------------------------------------------------------------------------- #
# Per-category builders (return list[case])
# --------------------------------------------------------------------------- #
def build_torque(conn, n: int) -> list[dict]:
    out = []
    for d in src.iter_torque_docs(conn):
        tq = first_torque(d.text_md or "")
        if not tq:
            continue
        veh = src.pick_vehicle_for_doc(d)
        out.append(_case("torque", len(out),
                         f"What's the tightening torque for {tq.component}?", veh,
                         {"gold_value": tq.value_nm, "gold_doc_ref": d.ref(),
                          "split_key": f"{veh.model or '?'}|torque"}))
        if len(out) >= n:
            break
    return out


def build_part(conn, n: int) -> list[dict]:
    out = []
    for p in src.iter_parts(conn):
        profs = src.profiles_for_part(conn, p.part_number, limit=3)
        veh = src.vehicle_from_profile(*profs[0]) if profs else gc.Vehicle()
        out.append(_case("part_number", len(out),
                         f"Look up Volvo part number {p.part_number} — what is it?", veh,
                         {"gold_part": p.part_number, "gold_desc": p.description,
                          "split_key": f"{veh.model or 'any'}|part"}))
        if len(out) >= n:
            break
    return out


def build_procedure(conn, n: int) -> list[dict]:
    out = []
    for d in src.iter_procedure_docs(conn):
        if not d.text_md or len(d.text_md) < 120:
            continue
        veh = src.pick_vehicle_for_doc(d)
        q = f"How do I {d.title[0].lower() + d.title[1:]}?" if d.title else "How do I service this?"
        out.append(_case("procedure", len(out), q, veh,
                         {"gold_doc_ref": d.ref(),
                          "gold_passage": " ".join((d.text_md or "").split())[:600],
                          "split_key": f"{veh.model or '?'}|procedure"}))
        if len(out) >= n:
            break
    return out


def build_pinout(conn, n: int) -> list[dict]:
    out = []
    for row in src.iter_pinouts(conn):
        if not row["function"]:
            continue
        veh = src.S60R_PINOUT_VEHICLE
        out.append(_case("pinout", len(out),
                         f"What is connected to {row['module']} pin {row['pin']}?", veh,
                         {"gold_text": row["function"], "gold_wire": row["wire_color"],
                          "split_key": "S60R|pinout"}))
        if len(out) >= n:
            break
    return out


def build_figure(conn, n: int) -> list[dict]:
    out = []
    for d in src.iter_figure_docs(conn):
        veh = src.pick_vehicle_for_doc(d)
        topic = (d.title or "this component")
        out.append(_case("figure", len(out), f"Show me the figure for {topic}.", veh,
                         {"gold_doc_ref": d.ref(), "split_key": f"{veh.model or 'any'}|figure"}))
        if len(out) >= n:
            break
    return out


def build_refusal(conn, n: int) -> list[dict]:
    """Out-of-corpus / unanswerable questions: the model MUST refuse, not invent."""
    out = []
    seeds = [
        ("What is the recommended tire pressure for a 2019 Tesla Model 3?", gc.Vehicle()),
        ("What's the cylinder firing order for a Ford Coyote V8?", gc.Vehicle()),
    ]
    for d in src.iter_procedure_docs(conn):
        veh = src.pick_vehicle_for_doc(d)
        # a real car, but a specific value not present for it
        out.append(_case("refusal", len(out),
                         "What's the exact tightening torque for the oil cooler banjo bolt?",
                         veh, {"expect_refusal": True, "example_type": "refusal",
                               "split_key": f"{veh.model or '?'}|refusal"}))
        if len(out) >= max(0, n - len(seeds)):
            break
    for q, veh in seeds:
        out.append(_case("refusal", len(out), q, veh,
                         {"expect_refusal": True, "example_type": "refusal",
                          "split_key": "out_of_corpus|refusal"}))
    return out[:n]


def build_cross_vehicle(conn, n: int) -> list[dict]:
    out = []
    for p in src.iter_parts(conn):
        profs = src.profiles_for_part(conn, p.part_number, limit=12)
        related = next((src.vehicle_from_profile(*t) for t in profs
                        if src.vehicle_from_profile(*t).engine and src.vehicle_from_profile(*t).model), None)
        if related is None:
            continue
        part_models = {src._first_token(m) for (_e, m, _y) in profs if m}
        active = src.sibling_vehicle_sharing_engine(conn, related.engine, part_models)
        if active is None:
            continue
        out.append(_case("cross_vehicle", len(out),
                         f"Is there a part number for the {p.description.lower()} on my car?",
                         active, {"gold_part": p.part_number, "expect_cross_vehicle": True,
                                  "example_type": "cross_vehicle",
                                  "related": related.label(),
                                  "split_key": f"{active.model}|cross_vehicle"}))
        if len(out) >= n:
            break
    return out


_BUILDERS = {
    "torque": build_torque, "part_number": build_part, "procedure": build_procedure,
    "pinout": build_pinout, "figure": build_figure, "refusal": build_refusal,
    "cross_vehicle": build_cross_vehicle,
}


def generate(conn, counts: dict[str, int]) -> list[dict]:
    cases: list[dict] = []
    for cat, fn in _BUILDERS.items():
        got = fn(conn, counts.get(cat, 0))
        cases.extend(got)
        print(f"  {cat:14} {len(got)}/{counts.get(cat, 0)}"
              + ("  <-- SHORT" if len(got) < counts.get(cat, 0) else ""))
    return cases


def _canon(case: dict) -> str:
    """The ONE canonical serialization of a case — used for BOTH the on-disk file and
    the content hash, so sha256(benchmark.jsonl bytes) reproduces the stored
    content_hash and 'frozen' is actually verifiable (see docs/AUDIT-2026-06-20.md)."""
    return json.dumps(case, ensure_ascii=False, sort_keys=True)


def _write_cases(cases: list[dict], out_path: str) -> str:
    """Write cases as canonical JSONL; return the exact bytes written (as str)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    blob = "".join(_canon(c) + "\n" for c in cases)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(blob)
    return blob


def freeze_artifacts(cases: list[dict], out_path: str) -> dict:
    """Write benchmark.jsonl + a content hash + frozen_split.json (split_keys +
    source_ids dataset/gen_dataset.py must exclude). The hash is taken over the EXACT
    bytes written, so a verifier can `sha256 benchmark.jsonl` and match content_hash."""
    blob = _write_cases(cases, out_path)
    content_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    split_keys = sorted({c["metadata"].get("split_key") for c in cases if c["metadata"].get("split_key")})
    source_ids = sorted({sid for c in cases
                         for sid in (c["metadata"].get("gold_doc_ref"),
                                     c["metadata"].get("gold_part")) if sid})
    frozen = {
        "version": "v1",
        "content_hash": content_hash,
        "n_cases": len(cases),
        "split_keys": split_keys,
        "source_ids": source_ids,
    }
    frozen_path = os.path.join(os.path.dirname(os.path.abspath(out_path)), "frozen_split.json")
    with open(frozen_path, "w", encoding="utf-8") as fh:
        json.dump(frozen, fh, indent=2)
    return {"content_hash": content_hash, "frozen_split": frozen_path,
            "split_keys": len(split_keys), "source_ids": len(source_ids)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build the frozen VIDA eval benchmark")
    ap.add_argument("--out", default="vida-eval/benchmark.jsonl")
    ap.add_argument("--kb", default=None)
    ap.add_argument("--freeze", action="store_true",
                    help="write content hash + frozen_split.json (exclude from training)")
    for cat, dflt in DEFAULT_COUNTS.items():
        ap.add_argument(f"--{cat}", type=int, default=dflt)
    args = ap.parse_args(argv)
    counts = {cat: getattr(args, cat) for cat in DEFAULT_COUNTS}

    conn = src.connect(args.kb)
    try:
        print("building benchmark:")
        cases = generate(conn, counts)
    finally:
        conn.close()

    if args.freeze:
        info = freeze_artifacts(cases, args.out)
        print(f"\nfroze {len(cases)} cases -> {args.out}")
        print(f"  content_hash {info['content_hash'][:16]}…  | "
              f"{info['split_keys']} split_keys, {info['source_ids']} source_ids excluded")
        print(f"  verify with: shasum -a 256 {args.out}  (matches content_hash in "
              f"{info['frozen_split']}); git-tag the hash so 'frozen' is enforceable")
    else:
        _write_cases(cases, args.out)
        print(f"\nwrote {len(cases)} cases -> {args.out} (not frozen; pass --freeze)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 1:
        p = src.db_path()
        if not os.path.exists(p):
            print(f"SKIP self-check: KB absent at {p}")
        else:
            conn = src.connect()
            try:
                cases = generate(conn, {k: 4 for k in DEFAULT_COUNTS})
            finally:
                conn.close()
            assert cases, "no cases"
            for c in cases:
                assert c["id"] and c["metadata"]["answer_type"], c
                assert c["messages"][0]["role"] == "system"

            # content_hash must be reproducible from the on-disk bytes (frozen is real)
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                out = os.path.join(td, "benchmark.jsonl")
                info = freeze_artifacts(cases, out)
                disk_hash = hashlib.sha256(open(out, "rb").read()).hexdigest()
                assert disk_hash == info["content_hash"], "content_hash != sha256(file bytes)"
                fz = json.load(open(info["frozen_split"]))
                assert fz["content_hash"] == disk_hash
            print(f"benchmark_gen.py self-check OK: {len(cases)} cases across "
                  f"{len({c['metadata']['answer_type'] for c in cases})} categories; "
                  f"content_hash reproducible from file bytes")
    else:
        raise SystemExit(main())
