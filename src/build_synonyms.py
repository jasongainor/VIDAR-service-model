"""Mine a synonym layer from VIDA's own en-US / en-GB Lexicon — no hand-curation.

The same EPC Lexicon DescriptionId carries en-US (mechanic-speak) and en-GB (VIDA's
wording). Where two descriptions differ in exactly ONE word position, that word pair is
a clean, domain-exact synonym (headlights↔headlamps, post↔pillar, module↔unit,
muffler↔silencer). We mine those MINIMAL-PAIR substitutions (high precision; multi-word
diffs are skipped) in BOTH directions and write them to a `synonyms` table in the store.
The server loads it and merges with the small hand-curated map, so query expansion covers
VIDA's own vocabulary variance for free.

Synonyms are VIDA-derived -> they live in the (gitignored) local store, never shipped.

Run AFTER the store exists, with vida-sql up:
  uv run python src/build_synonyms.py
"""
import difflib
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "vida-kb.sqlite3"
EN_US, EN_GB = 15, 16

_WORD = re.compile(r"[a-z]+")
# noise: VIDA internal markers / positional abbreviations that aren't synonyms
_DROP_TOKENS = {"gcp", "lh", "rh", "lhd", "rhd", "l", "r"}
# never map these generic words (would distort ranking); keep synonyms meaningful
_STOPish = {"the", "and", "for", "with", "left", "right", "front", "rear", "upper",
            "lower", "inner", "outer", "complete", "assy", "assembly", "kit", "set"}


def _env(key: str) -> str | None:
    if key in os.environ:
        return os.environ[key]
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return None


def _tokens(desc: str) -> list[str]:
    return [t for t in _WORD.findall((desc or "").lower())]


def _minimal_pair(a: list[str], b: list[str]):
    """If a and b differ in exactly one aligned position, return (a_word, b_word)."""
    if len(a) != len(b) or not a:
        return None
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    if len(diffs) != 1:
        return None
    return diffs[0]


def _clean_span(tokens: list[str]) -> list[str] | None:
    """Reject a span carrying VIDA internal markers / pure stop-words (would be noise)."""
    if not tokens or any(t in _DROP_TOKENS for t in tokens):
        return None
    if all(t in _STOPish for t in tokens):
        return None
    return tokens


def _phrase_subst(a: list[str], b: list[str]):
    """Item-10 v2: the ONE contiguous span where a and b differ, when the rest aligns and
    the span is NOT a clean single-word↔single-word pair (those go to `synonyms`). Catches
    multi-word and UNEQUAL-length variants the minimal-pair miner can't — 'gear box'↔
    'gearbox', 'anti roll bar'↔'sway bar'. Returns (a_phrase, b_phrase) or None."""
    if not a or not b:
        return None
    ops = [op for op in difflib.SequenceMatcher(None, a, b).get_opcodes() if op[0] != "equal"]
    if len(ops) != 1 or ops[0][0] != "replace":  # exactly one contiguous substitution
        return None
    _, i1, i2, j1, j2 = ops[0]
    sa, sb = _clean_span(a[i1:i2]), _clean_span(b[j1:j2])
    if not sa or not sb or (len(sa) == 1 and len(sb) == 1):  # single-word pair -> `synonyms`
        return None
    pa, pb = " ".join(sa), " ".join(sb)
    return None if pa == pb else (pa, pb)


def main():
    t0 = time.time()
    user, pw = _env("VIDA_SQL_USER"), _env("VIDA_SQL_PASSWORD")
    if not user or not pw:
        sys.exit("VIDA_SQL_USER / VIDA_SQL_PASSWORD missing (.env)")
    import pytds

    con = pytds.connect(server="127.0.0.1", port=1433, database="EPC", user=user,
                        password=pw, autocommit=True, login_timeout=5, timeout=60)
    cur = con.cursor()
    cur.execute(
        "SELECT us.Description, gb.Description FROM dbo.Lexicon us"
        " JOIN dbo.Lexicon gb ON us.DescriptionId = gb.DescriptionId"
        " WHERE us.fkLanguage = %s AND gb.fkLanguage = %s AND us.Description <> gb.Description",
        (EN_US, EN_GB),
    )
    pairs = cur.fetchall()
    con.close()

    # term -> Counter of alternative terms (both directions)
    votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # phrase -> Counter of alternative phrases (item 10 v2: multi-word / unequal length)
    pvotes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    considered = 0
    for us, gb in pairs:
        ta, tb = _tokens(us), _tokens(gb)
        mp = _minimal_pair(ta, tb)
        if mp:
            a, b = mp
            if a != b and len(a) >= 3 and len(b) >= 3 \
                    and a not in _DROP_TOKENS and b not in _DROP_TOKENS \
                    and a not in _STOPish and b not in _STOPish:
                considered += 1
                votes[a][b] += 1
                votes[b][a] += 1
            continue
        pp = _phrase_subst(ta, tb)
        if pp:
            pa, pb = pp
            pvotes[pa][pb] += 1
            pvotes[pb][pa] += 1

    # keep an alternative only if it shows up at least twice (drops one-off typos like
    # "accelerometerr"), capped per term so no single word explodes the OR-group
    syn: dict[str, list[str]] = {}
    for term, alts in votes.items():
        kept = sorted([w for w, n in alts.items() if n >= 2], key=lambda w: -alts[w])[:6]
        if kept:
            syn[term] = kept

    # phrase synonyms: keep an alternative seen >=2x, capped per phrase (same precision bar)
    phrases: dict[str, list[str]] = {}
    for phrase, alts in pvotes.items():
        kept = sorted([w for w, n in alts.items() if n >= 2], key=lambda w: -alts[w])[:4]
        if kept:
            phrases[phrase] = kept

    lite = sqlite3.connect(DB_PATH)
    lite.execute("CREATE TABLE IF NOT EXISTS synonyms (term TEXT, alt TEXT, n INTEGER,"
                 " PRIMARY KEY (term, alt))")
    lite.execute("CREATE TABLE IF NOT EXISTS phrase_synonyms (phrase TEXT, alt TEXT, n INTEGER,"
                 " PRIMARY KEY (phrase, alt))")
    lite.execute("DELETE FROM synonyms")
    lite.execute("DELETE FROM phrase_synonyms")
    rows = [(term, alt, votes[term][alt]) for term, alts in syn.items() for alt in alts]
    prows = [(p, alt, pvotes[p][alt]) for p, alts in phrases.items() for alt in alts]
    lite.executemany("INSERT OR REPLACE INTO synonyms VALUES (?,?,?)", rows)
    lite.executemany("INSERT OR REPLACE INTO phrase_synonyms VALUES (?,?,?)", prows)
    lite.commit()
    lite.close()

    print(f"lexicon pairs: {len(pairs)} | minimal-pair subs: {considered} | "
          f"terms with synonyms: {len(syn)} | rows written: {len(rows)} | "
          f"phrase terms: {len(phrases)} | phrase rows: {len(prows)} | "
          f"{time.time()-t0:.0f}s", flush=True)
    for term, alts in sorted(syn.items())[:20]:
        print(f"  {term:18} -> {alts}")
    print("  -- phrase synonyms --")
    for p, alts in sorted(phrases.items())[:20]:
        print(f"  {p:24} -> {alts}")


if __name__ == "__main__":
    main()
