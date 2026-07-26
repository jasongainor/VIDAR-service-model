"""vida-eval retrieval scorecard — deterministic, LLM-free ranking checks.

Where run.py drives the whole LLM tool-loop (slow, needs a model loaded), this
scorecard imports the MCP server in-process and calls search_docs() directly, then
asserts RANKING expectations: the right document is retrievable, near the top, above
a known-wrong one, or absent. It is the measurement instrument for retrieval tuning —
fusion weights (item 1), lexical mis-ranks (item 2), embedding precision (item 3) and
conflict flagging (item 8). Fast enough to run after every retrieval change.

Ground truth is the same human-verified set as cases.json (VCC refs / values verified
against the store during acceptance). Each check explains what it guards.

Usage:
  uv run python vida-eval/retrieval.py            # full scorecard, exit 1 on any fail
  uv run python vida-eval/retrieval.py --show      # also print top-6 per query (debug)
  VIDA_DISABLE_VECTORS=1 uv run python vida-eval/retrieval.py   # pure-FTS (isolate lexical)

Exit code: 0 = all checks pass, 1 = any failure.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import server  # noqa: E402


def _ref(r):
    return (r.get("vida_doc_ref") or "")


def _title(r):
    return (r.get("title") or "")


_ABSENT = 10**9  # rank for "not found" — must NOT collide with a real index (esp. 0)


def rank_of(results, *, ref=None, title_re=None):
    """Index of the first result matching a VCC ref prefix or a title regex, else None."""
    for i, r in enumerate(results):
        if ref and _ref(r).upper().startswith(ref.upper()):
            return i
        if title_re and re.search(title_re, _title(r), re.I):
            return i
    return None


def pos(results, *, ref=None, title_re=None):
    """rank_of() but None -> _ABSENT, so `pos(...) < n` is correct even at index 0
    (the `rank_of(...) or 99` idiom silently treats a #0 hit as absent)."""
    r = rank_of(results, ref=ref, title_re=title_re)
    return _ABSENT if r is None else r


def run_query(q, profile=None, limit=8):
    res = server.search_docs(q, profile=profile, limit=limit)
    # the first item may carry a 'notice' wrapper but still be a real result
    return res


# ── Checks ────────────────────────────────────────────────────────────────────
# Each check: (id, description, fn(results)->None|failure_str). Queries are grouped so
# one search feeds several assertions.

def _rank_above(rs, ref_a, ref_b):
    ra, rb = rank_of(rs, ref=ref_a), rank_of(rs, ref=ref_b)
    if ra is None:
        return f"{ref_a} not present (cannot outrank {ref_b})"
    if rb is not None and ra > rb:
        return f"{ref_a} ranked #{ra} BELOW {ref_b} #{rb}"
    return None


def _has_head_doc_in_top(rs, n):
    for r in rs[:n]:
        t = _title(r)
        if re.search(r"cylinder head|cylinder block|group 2[01]\b", t, re.I):
            return True
    return False


def _no_wrong_engine_lead(rs):
    for r in rs:
        if r.get("scope", "").startswith("other engine") and r.get("lead"):
            return f"wrong-engine doc {_ref(r)} leaked a quotable lead"
    return None


CASES = [
    {
        "q": "intake manifold tightening torque M6 bolts",
        "why": "M6 conflict (item 2/8): the this-car 19 Nm spec (VCC-112365) must rank "
               "above the other-vehicle 20 Nm doc (VCC-131554), which previously won.",
        "checks": [
            ("m6-this-car-present", lambda rs: None if rank_of(rs, ref="VCC-112365") is not None
                else "VCC-112365 (this-car 19 Nm spec) not retrieved at all"),
            ("m6-this-car-beats-other", lambda rs: _rank_above(rs, "VCC-112365", "VCC-131554")),
            ("m6-conflict-flagged", lambda rs: None if any("conflict" in k for r in rs for k in r)
                else "no conflict_warning surfaced despite 19 vs 20 Nm disagreement"),
        ],
    },
    {
        "q": "intake manifold tightening torque",
        "why": "Plain phrasing must also surface the this-car spec table (VCC-112365 / 19 Nm) "
               "near the top — it carries the authoritative value.",
        "checks": [
            ("intake-plain-spec-top5", lambda rs: None if pos(rs, ref="VCC-112365") < 5
                else "VCC-112365 not in top 5 for plain intake-torque query"),
        ],
    },
    {
        "q": "cylinder head bolt torque",
        "why": "Lexical mis-rank (item 2): 'Intake manifold, replacement' (VCC-131554) merely "
               "contains all four words and used to rank #0; a cylinder-head doc must beat it.",
        "checks": [
            ("cylhead-intake-not-top2", lambda rs: None if pos(rs, ref="VCC-131554") >= 2
                else "intake-manifold doc (VCC-131554) is in the top 2 for a cylinder-head query"),
            ("cylhead-headdoc-top3", lambda rs: None if _has_head_doc_in_top(rs, 3)
                else "no cylinder-head/Group 21/Group 20 doc in the top 3"),
        ],
    },
    {
        "q": "cylinder head bolt torque sequence",
        "why": "Regression guard: the discriminating word 'sequence' already converges on the "
               "this-car cylinder-block doc — keep it in the top 2.",
        "checks": [
            ("cylhead-seq-top2", lambda rs: None if _has_head_doc_in_top(rs, 2)
                else "cylinder-head doc fell out of the top 2 when 'sequence' was added"),
        ],
    },
    {
        "q": "key won't turn",
        "why": "Embedding precision (item 3): a lexical-poor query has no strict lexical match, "
               "so the loose fallback used to surface 'Ski holder' at #0. The fix REPLACES that "
               "loose junk with confident vector hits ('Key warning'), so it needs the vector layer.",
        "needs_vectors": True,
        "checks": [
            ("noise-no-ski-top3", lambda rs: None if pos(rs, title_re=r"ski holder|snowboard holder") >= 3
                else "'Ski/Snowboard holder' is in the top 3 for 'key won't turn' (noise)"),
        ],
    },
    {
        "q": "shuddering when accelerating",
        "why": "Semantic-recall regression guard (item 1/3): the vector layer must still reach "
               "'Checking clutch shudder' from a colloquial phrasing.",
        "checks": [
            ("semantic-clutch-top3", lambda rs: None if pos(rs, title_re=r"clutch shudder") < 3
                else "'Checking clutch shudder' dropped out of the top 3 (lost semantic recall)"),
        ],
    },
    {
        "q": "VCC-137506-1",
        "why": "Direct VCC reference must resolve to that exact document.",
        "checks": [
            ("vcc-direct-resolves", lambda rs: None if rank_of(rs, ref="VCC-137506") == 0
                else "VCC-137506-1 did not resolve to the top result"),
        ],
    },
    {
        "q": "glow plug resistance",
        "why": "Engine-scope guard (mirrors test_engine_scope): a diesel-only doc may rank but "
               "must never hand its quotable lead text on the default car scope.",
        "checks": [
            ("glowplug-no-wrong-lead", _no_wrong_engine_lead),
        ],
    },
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", action="store_true", help="print top-6 results per query")
    args = ap.parse_args()

    total = passed = skipped = 0
    failures = []
    vectors_on = getattr(server, "_VECTORS_ENABLED", True)
    for case in CASES:
        if case.get("needs_vectors") and not vectors_on:
            print(f"=== {case['q']!r}  (skipped — needs the vector layer; VIDA_DISABLE_VECTORS set)")
            skipped += len(case["checks"])
            continue
        rs = run_query(case["q"])
        print(f"=== {case['q']!r}")
        if args.show:
            for i, r in enumerate(rs[:6]):
                if "notice" in r:
                    print(f"      (notice: {r['notice'][:70]})")
                print(f"    {i}. [{r.get('scope','?')}] {_title(r)!r} {_ref(r)}")
        for cid, fn in case["checks"]:
            total += 1
            try:
                err = fn(rs)
            except Exception as e:  # a check should never crash the run
                err = f"check raised {e!r}"
            if err:
                failures.append((cid, err))
                print(f"  FAIL {cid}: {err}")
            else:
                passed += 1
                print(f"  pass {cid}")
        print()

    tail = f" ({skipped} skipped — vector layer disabled)" if skipped else ""
    print(f"{passed}/{total} retrieval checks passed{tail}")
    if failures:
        print("\nFAILURES:")
        for cid, err in failures:
            print(f"  - {cid}: {err}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
