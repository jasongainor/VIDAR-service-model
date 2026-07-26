"""Incremental-rebuild logic (item 5) — no SQL Server needed.

extract_docs.py can re-run cheaply: it hashes each document's SOURCE blob and skips
re-parsing byte-identical docs, re-parses changed ones, and prunes rows whose source
doc is gone. This exercises the pure decision helpers that drive that, plus a simulated
incremental pass over an in-memory store, so the logic is guarded without a live DB.

Run:  uv run python tests/test_incremental.py
"""
import importlib.util
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("extract_docs", ROOT / "src" / "extract_docs.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

failures = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


print("content hash:")
check("stable for identical bytes", ex._doc_hash(b"abc") == ex._doc_hash(b"abc"))
check("differs for changed bytes", ex._doc_hash(b"abc") != ex._doc_hash(b"abd"))
check("short hex digest", len(ex._doc_hash(b"x")) == 16)

print("\nplan_prune:")
check("incremental prunes vanished docs", ex.plan_prune({"a", "b", "c"}, {"a", "b"}, full=False) == {"c"})
check("incremental keeps still-present docs", ex.plan_prune({"a", "b"}, {"a", "b", "x"}, full=False) == set())
check("full mode prunes nothing here (table wiped up front)", ex.plan_prune({"a", "b"}, {"a"}, full=True) == set())

print("\nsimulated incremental pass (skip unchanged / refresh changed / add new / prune gone):")
# in-memory store mirroring the doc_hashes ledger + a documents table
con = sqlite3.connect(":memory:")
con.executescript("CREATE TABLE documents (doc_id TEXT PRIMARY KEY, text_md TEXT);"
                  "CREATE TABLE doc_hashes (doc_id TEXT PRIMARY KEY, hash TEXT);")
# first run: three docs
source_v1 = {"d1": b"one", "d2": b"two", "d3": b"three"}
for did, blob in source_v1.items():
    con.execute("INSERT INTO documents VALUES (?,?)", (did, blob.decode()))
    con.execute("INSERT INTO doc_hashes VALUES (?,?)", (did, ex._doc_hash(blob)))
con.commit()

# second run: d1 unchanged, d2 changed, d3 gone, d4 new
source_v2 = {"d1": b"one", "d2": b"two-EDIT", "d4": b"four"}
seen = dict(con.execute("SELECT doc_id, hash FROM doc_hashes"))
parsed, skipped = [], []
for did, blob in source_v2.items():
    h = ex._doc_hash(blob)
    if seen.get(did) == h:
        skipped.append(did)
        continue
    parsed.append(did)
    con.execute("INSERT OR REPLACE INTO documents VALUES (?,?)", (did, blob.decode()))
    con.execute("INSERT OR REPLACE INTO doc_hashes VALUES (?,?)", (did, h))
for did in ex.plan_prune(set(seen), set(source_v2), full=False):
    con.execute("DELETE FROM documents WHERE doc_id=?", (did,))
    con.execute("DELETE FROM doc_hashes WHERE doc_id=?", (did,))
con.commit()

check("d1 skipped (unchanged)", skipped == ["d1"])
check("d2 + d4 re-parsed (changed + new)", set(parsed) == {"d2", "d4"})
final = {r[0]: r[1] for r in con.execute("SELECT doc_id, text_md FROM documents")}
check("final store has d1,d2,d4 only (d3 pruned)", set(final) == {"d1", "d2", "d4"})
check("d2 content refreshed", final["d2"] == "two-EDIT")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    raise SystemExit(1)
print("All incremental-rebuild checks passed.")
