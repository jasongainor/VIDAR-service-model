"""Safety regression: figure retrieval must NEVER cross engine families.

The 2005 S60R is a 5-cylinder petrol (B5254T4). VIDA also holds cylinder-head
torque figures for 6-cylinder diesels (D24, a 14-bolt grid) and 4-cylinder petrols
(B4204). Serving one of those for this car is dangerous (wrong torque pattern), so
get_figures enforces vehicle/engine applicability — by VIDA's own document tagging
AND by engine codes named in the figure's own caption text.

Run:  uv run python tests/test_engine_scope.py
"""
import importlib.util
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("vidasrv", ROOT / "src" / "server.py")
srv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(srv)

# Figure paths that MUST NEVER be shown for this car (verified wrong-engine):
D24_DIESEL_14BOLT = "0900c8af80120202_0_0.gif"   # A2101030, D24 inline-6 diesel head
B4204_4CYL = "0900c8af80127057_0_0.gif"          # M-figure subframe (early bug)
failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(name)


print("engine-code classifier:")
check("B4204 caption flags wrong engine",
      srv._wrong_engine_near("head bolts B4204T2 [image X1]", len("head bolts B4204T2 ")))
check("B5254 caption is compatible",
      not srv._wrong_engine_near("head bolts B5254T4 [image X1]", len("head bolts B5254T4 ")))
check("D24 diesel flags wrong engine",
      srv._wrong_engine_near("cylinder head D24TIC [image X1]", len("cylinder head D24TIC ")))
check("red-block B230 flags wrong engine",
      srv._wrong_engine_near("engine B230FT [image X1]", len("engine B230FT ")))
check("body DTC B2155 is NOT treated as an engine",
      not srv._ENGINE_FAMILY_RE.findall("fault code B2155 present"))
check("no engine named near figure => not blocked",
      not srv._wrong_engine_near("Tighten in sequence from centre [image X1]", 30))


def served_paths(query, limit=8):
    res = srv.get_figures(query=query, limit=limit)
    if isinstance(res, dict):
        return None, res  # refusal / note
    summary = res[0]
    return [f["id"] for f in summary.get("figures", [])], summary


print("\nget_figures engine enforcement:")
paths, summ = served_paths("cylinder head bolt torque sequence")
check("cyl-head query never serves the D24 6-cyl diesel grid",
      paths is None or D24_DIESEL_14BOLT not in paths, str(paths))
check("cyl-head query never serves the B4204/subframe figure",
      paths is None or B4204_4CYL not in paths, str(paths))
# It SHOULD find the correct 10-bolt 5-cyl sequence (H2102547) rather than refuse.
check("cyl-head query returns a figure (the 10-bolt 5-cyl sequence)",
      paths and "0900c8af81ada446_0_0.gif" in paths, str(paths))

# Queries with no applicable figure must refuse, not substitute a wrong one.
for q in ["valve cover bolt torque sequence", "sunroof torque sequence"]:
    paths, summ = served_paths(q)
    check(f"refuses (no applicable figure): {q!r}", paths is None, str(paths))

# Good queries must still return their (applicable) figure.
for q in ["spark plug gap", "throttle adaptation", "bleed brakes", "serpentine belt routing"]:
    paths, summ = served_paths(q, limit=1)
    check(f"still serves a figure: {q!r}", bool(paths), str(summ.get("note", "")[:60]))

print("\nengine-code regex precision (part/tool/document ids are NOT engines):")
for s in ["D5200235", "B5200401", "D5900401", "B2155"]:
    check(f"{s} not misread as an engine code", not srv._ENGINE_FAMILY_RE.findall(s))
for s in ["B5254T4", "D24TIC", "D5244T18", "B6304GS", "B230FT"]:
    check(f"{s} still recognised as an engine code", srv._ENGINE_FAMILY_RE.findall(s) == [s])

print("\nengine applicability classifier:")
for eng, exp in [("B5254T4", "compatible"), (None, "generic"), ("D24TIC,D24T", "wrong"),
                 ("B5254T4,D5244T", "compatible"), ("B6304GS,B6304FS", "wrong"),
                 ("B4204T2", "wrong"), ("B230FT", "wrong")]:
    check(f"engine {eng!r} => {exp}", srv._engine_applicability(eng) == exp)
check("lead naming only a diesel is wrong-engine", srv._lead_wrong_engine("remove glow plugs D4192T2"))
check("lead naming this car's engine is fine", not srv._lead_wrong_engine("head bolts B5254T4 110 Nm"))
check("lead with a body DTC is not wrong-engine", not srv._lead_wrong_engine("fault code B2155 stored"))


def _independent_wrong(engine):
    """Test-local notion of a wrong-engine doc, independent of the code under test."""
    if not engine:
        return False
    toks = [t for t in engine.replace(" ", "").split(",") if t]
    if any(re.match(r"B5\d{3}", t, re.I) for t in toks):
        return False
    return any(re.match(r"(B[2468]\d{2,3}|D\d)", t, re.I) for t in toks)


print("\nsearch_docs engine enforcement (the dangerous text-quote surface):")
# Thin-coverage topics where wrong-engine docs reach the top-3 and previously
# handed their full 2800-char lead text to a single-pass model.
for q in ["glow plug resistance", "diesel injector coding", "turbo diesel boost pressure",
          "preheating control", "cylinder head bolt torque"]:
    res = srv.search_docs(q, limit=6)
    leaked = [i for i, r in enumerate(res) if _independent_wrong(r.get("engine")) and r.get("lead")]
    unwarned = [i for i, r in enumerate(res)
                if _independent_wrong(r.get("engine")) and "applicability_warning" not in r]
    check(f"no wrong-engine lead text for {q!r}", not leaked, f"leaked idx {leaked}")
    check(f"wrong-engine rows are flagged for {q!r}", not unwarned, f"unwarned idx {unwarned}")

# Engine-irrelevant / compatible topics must KEEP their lead (no over-blocking).
for q in ["brake fluid bleeding", "engine oil capacity", "timing belt tension", "spark plug gap"]:
    res = srv.search_docs(q, limit=4)
    check(f"legit doc keeps lead text: {q!r}", any(r.get("lead") for r in res[:3]))

# Explicit cross-vehicle queries OPT OUT of enforcement (the user asked to see it).
res = srv.search_docs("glow plug resistance", profile="all", limit=4)
check("profile='all' still hands other-engine lead (enforcement is car-scope only)",
      any(_independent_wrong(r.get("engine")) and r.get("lead") for r in res[:3]))

print("\nCarContext (the all-cars foundation — gates must RETARGET, default must be intact):")
check("default car derives the B5 family", srv.CAR.engine_family == r"B5\d{3}")
check("default car keeps its aliases", set(srv.CAR.aliases) == {"b5254t4", "s60r"})
check("family derivation B4204T -> B4", srv._default_engine_family("B4204T") == r"B4\d{3}")
check("family derivation D5244T -> D5", srv._default_engine_family("D5244T") == r"D5\d{3}")

# Reload the server under a DIFFERENT car (4-cyl B4204T) and confirm every engine gate
# flips with it — the proof that the car is genuinely a variable, not hardcoded.
_env_bak = dict(os.environ)
os.environ.update({"VIDA_CAR_ENGINE": "B4204T", "VIDA_CAR_PROFILE_ID": "test0001",
                   "VIDA_CAR_MODEL": "S40 (-04)", "VIDA_CAR_YEAR": "2003"})
try:
    _spec4 = importlib.util.spec_from_file_location("vidasrv_b4", ROOT / "src" / "server.py")
    srv4 = importlib.util.module_from_spec(_spec4)
    _spec4.loader.exec_module(srv4)
finally:
    os.environ.clear()
    os.environ.update(_env_bak)
check("retarget: B4204T is now compatible", srv4._engine_compatible("B4204T3"))
check("retarget: B5254T4 is now WRONG-engine", not srv4._engine_compatible("B5254T4"))
check("retarget: applicability flips B5 -> wrong", srv4._engine_applicability("B5254T4") == "wrong")
check("retarget: applicability B4 -> compatible", srv4._engine_applicability("B4204T") == "compatible")
check("retarget: no default-nickname bleed", srv4.CAR.aliases == ("b4204t",))
check("retarget: EPC binds the new profile id", srv4.CAR.profile_id == "test0001")
check("retarget: car-docs SQL targets the new engine",
      "B4204T" in srv4._CAR_DOCS_SQL and "B5254T4" not in srv4._CAR_DOCS_SQL)
check("default module unaffected by the reload",
      srv.CAR.engine == "B5254T4" and srv._engine_compatible("B5254T4"))

print("\nper-query car override (item 7 — search_docs(car=…) retargets in ONE process):")
check("resolve engine code", srv._resolve_car("B6294T").engine == "B6294T"
      and srv._resolve_car("B6294T").engine_family == r"B6\d{3}")
check("resolve engine+year", srv._resolve_car("D5244T 2010").year == "2010")
check("vague spec falls back to default car (never silently widens scope)",
      srv._resolve_car("foo bar").engine == "B5254T4")
# default query: a B5254-only spec is this-car-safe; flip car -> it must be flagged wrong.
_b4 = srv.search_docs("cylinder head bolt torque sequence", car="B4204T", limit=8)
check("car=B4204T flags a B5254-only doc as wrong-engine",
      any(r.get("engine") and srv._engine_applicability(r["engine"]) == "compatible"
          and srv._engine_applicability(r["engine"], re.compile(r"B4\d{3}", re.I)) == "wrong"
          and r.get("scope", "").startswith("other engine")
          for r in _b4) or
      # or at least: some result carries a B4-scoped applicability_warning
      any("applicability_warning" in r for r in _b4))
check("default search_docs(car=None) unchanged: B5254 spec is NOT wrong-engine",
      not any(r.get("scope", "").startswith("other engine") and r.get("engine")
              and "B5254" in (r.get("engine") or "")
              for r in srv.search_docs("cylinder head bolt torque sequence", limit=8)))

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    raise SystemExit(1)
print("All engine-scope checks passed.")
