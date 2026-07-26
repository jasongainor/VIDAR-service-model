"""CGM Milestone 2 (item 9) — converter + SVG serving, no Java needed.

VIDA's vector diagrams are CGM; the document corpus references only a handful (≈1).
extract_cgm.py renders them to SVG via an external converter and bakes them into the
image store under a `<path>.svg` key; the server serves that SVG when a doc references
the `.cgm`. This exercises the path-derivation, the convert step (with a stub converter
standing in for jcgm), and the server's CGM->SVG resolution + SVG serving — without a
JDK and without writing into the real image store.

Run:  uv run python tests/test_cgm.py
"""
import importlib.util
import sqlite3
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


srv = _load("vidasrv_cgm", ROOT / "src" / "server.py")
ec = _load("extract_cgm", ROOT / "src" / "extract_cgm.py")

failures = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


CGM = "0900c8af834c8fad_180_263.cgm"
SVG = "0900c8af834c8fad_180_263.svg"

PNG = "0900c8af834c8fad_180_263.png"

print("path derivation + server format support:")
check("rendered_path_for swaps .cgm -> .png (jcgm-core raster)", ec.rendered_path_for(CGM) == PNG)
check("rendered_path_for can target .svg", ec.rendered_path_for(CGM, ".svg") == SVG)
check("server knows the .svg format", srv._IMG_FORMAT.get(".svg") == "svg")
check("image server path regex accepts .svg", bool(srv._IMAGE_PATH_RE.fullmatch(SVG)))
check("image server path regex still rejects junk", not srv._IMAGE_PATH_RE.fullmatch("../etc/passwd"))

print("\nconvert step (stub converter stands in for jcgm):")
svg_bytes = ec.convert(b"<cgm-bytes>", "printf '%s' '<svg xmlns=\"http://www.w3.org/2000/svg\"/>' > {out}")
check("converter produces SVG bytes", svg_bytes is not None and svg_bytes.startswith(b"<svg"))
check("converter returns None when it writes nothing", ec.convert(b"x", "true") is None)

print("\nserver serves a baked SVG for a referenced CGM (URL mode = the deployment path):")
with tempfile.TemporaryDirectory() as d:
    store = Path(d) / "img.sqlite3"
    con = sqlite3.connect(store)
    con.execute("CREATE TABLE images (path TEXT PRIMARY KEY, content_type TEXT, data BLOB)")
    con.execute("INSERT INTO images VALUES (?,?,?)", (SVG, "image/svg+xml", svg_bytes))
    con.commit()
    con.close()
    img_con = sqlite3.connect(f"file:{store}?mode=ro", uri=True)

    # base_url set => figures serve over HTTP; a referenced .cgm must resolve to its baked .svg
    images, shown, skipped = srv._fetch_figures([CGM], 4, "http://host:8766", img_con, None)
    check("CGM with a baked SVG is shown (not skipped)", len(shown) == 1 and not skipped)
    check("served URL points at the .svg sibling", shown and shown[0]["url"].endswith(SVG))
    check("markdown image line emitted", shown and shown[0]["markdown"].startswith("![Figure"))

    # a referenced .cgm with NO baked SVG degrades to a labelled skip, not an error
    images2, shown2, skipped2 = srv._fetch_figures(["deadbeef_1_2.cgm"], 4, "http://host:8766", img_con, None)
    check("CGM without a baked SVG degrades to a note", not shown2 and len(skipped2) == 1
          and "CGM" in skipped2[0]["reason"])
    img_con.close()

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    raise SystemExit(1)
print("All CGM Milestone-2 checks passed.")
