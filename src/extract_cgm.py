"""CGM Milestone 2: render VIDA's WebCGM vector diagrams to SVG and bake them locally.

VIDA stores wiring/diagnostic diagrams as WebCGM v4 vectors (~14% of the whole image
repository, but only a HANDFUL are referenced by the indexed service documents — almost
all document figures are raster, handled by extract_images.py). This script renders the
CGMs a document actually references into SVG and bakes them into the SAME sidecar store
(data/vida-images.sqlite3) under a `<path>.svg` key. The server then serves the SVG when
a document references the `.cgm` (get_figures / the HTTP image server already handle it).

Rendering uses an EXTERNAL converter you provide (jcgm renders ~96% of VIDA CGMs — see
docs/images-plan.md for the audited jcgm-core + jcgm-to-svg recipe). Point this at it:

  VIDA_CGM_CMD='java -cp jcgm-core.jar:jcgm-to-svg.jar net.sf.jcgm.SVGConverter {in} {out}'

`{in}` / `{out}` are substituted with temp file paths. If VIDA_CGM_CMD is unset and no
`java` is on PATH, the script bakes nothing and prints exactly what to install — it never
fails the build (CGMs degrade to a labelled note in the UI, as before).

Run (with vida-sql up for the CGM bytes, or --svg-dir for pre-rendered SVGs):
  VIDA_CGM_CMD='…' uv run python src/extract_cgm.py
  uv run python src/extract_cgm.py --svg-dir /path/to/prerendered   # bake existing SVGs
"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "vida-kb.sqlite3"
IMAGES_DB = ROOT / "data" / "vida-images.sqlite3"

# The converter may emit SVG (jcgm-to-svg) or a raster (jcgm-core -> PNG). We sniff the
# output's magic bytes and bake under a sibling with the matching extension/content-type,
# so the server serves whatever the renderer produced.
_SNIFF = [
    (b"\x89PNG\r\n\x1a\n", ".png", "image/png"),
    (b"GIF8", ".gif", "image/gif"),
    (b"\xff\xd8\xff", ".jpeg", "image/jpeg"),
]


def _sniff(data: bytes) -> tuple[str, str]:
    """(extension, content_type) for rendered bytes — SVG if it looks like XML, else raster."""
    for magic, ext, ct in _SNIFF:
        if data.startswith(magic):
            return ext, ct
    head = data[:200].lstrip().lower()
    if head.startswith(b"<?xml") or head.startswith(b"<svg"):
        return ".svg", "image/svg+xml"
    return ".png", "image/png"  # default assumption


def _env(key: str) -> str | None:
    if key in os.environ:
        return os.environ[key]
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return None


def rendered_path_for(cgm_path: str, ext: str = ".png") -> str:
    """The baked key the server looks up for a referenced CGM (extension swapped to the
    rendered format). Default .png (jcgm-core raster); .svg when a vector converter is used."""
    stem = cgm_path[:-4] if cgm_path.lower().endswith(".cgm") else cgm_path
    return stem + ext


# kept for backwards-compatibility with the test that named the SVG path explicitly
def svg_path_for(cgm_path: str) -> str:
    return rendered_path_for(cgm_path, ".svg")


def referenced_cgms() -> list[str]:
    """Every .cgm path referenced by an indexed document."""
    lite = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    refs: set[str] = set()
    for (ir,) in lite.execute(
        "SELECT image_refs FROM documents WHERE image_refs LIKE '%.cgm%'"
    ):
        for p in json.loads(ir or "[]"):
            if p.lower().endswith(".cgm"):
                refs.add(p)
    lite.close()
    return sorted(refs)


_REND_EXTS = (".png", ".svg", ".gif", ".jpeg")


def convert(cgm_bytes: bytes, cmd_template: str) -> bytes | None:
    """Render CGM bytes via the external converter (SVG or raster), or None on failure.
    The output format is whatever the converter writes — sniffed at bake time."""
    with tempfile.TemporaryDirectory() as d:
        cin, cout = os.path.join(d, "in.cgm"), os.path.join(d, "out")
        Path(cin).write_bytes(cgm_bytes)
        cmd = cmd_template.format(**{"in": cin, "out": cout})
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, timeout=180)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            err = getattr(e, "stderr", b"") or b""
            print(f"  converter failed: {err[-180:].decode('utf-8', 'replace')}", file=sys.stderr)
            return None
        if os.path.exists(cout) and os.path.getsize(cout) > 0:
            return Path(cout).read_bytes()
        return None


def _already_baked(cgm: str, have: set[str]) -> bool:
    return any(rendered_path_for(cgm, e) in have for e in _REND_EXTS)


def open_images_store() -> sqlite3.Connection:
    out = sqlite3.connect(IMAGES_DB)
    out.execute("PRAGMA journal_mode=WAL")
    out.execute("CREATE TABLE IF NOT EXISTS images ("
                " path TEXT PRIMARY KEY, content_type TEXT, data BLOB)")
    out.commit()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render-dir", help="bake files already rendered elsewhere (named "
                                         "<cgm-stem>.png/.svg) instead of invoking a converter")
    ap.add_argument("--force", action="store_true", help="re-bake even if a render already exists")
    args = ap.parse_args()
    t0 = time.time()

    cgms = referenced_cgms()
    print(f"document-referenced CGMs: {len(cgms)}", flush=True)
    if not cgms:
        print("nothing to render — the document corpus references no CGM figures.")
        return

    out = open_images_store()
    have = {r[0] for r in out.execute("SELECT path FROM images")}
    baked = skipped = 0

    if args.render_dir:  # offline path: bake files already rendered elsewhere
        d = Path(args.render_dir)
        for cgm in cgms:
            if _already_baked(cgm, have) and not args.force:
                skipped += 1
                continue
            for ext in _REND_EXTS:
                src = d / Path(rendered_path_for(cgm, ext)).name
                if src.exists():
                    data = src.read_bytes()
                    e, ct = _sniff(data)
                    sp = rendered_path_for(cgm, e)
                    out.execute("INSERT OR REPLACE INTO images VALUES (?,?,?)", (sp, ct, data))
                    have.add(sp)
                    baked += 1
                    break
        out.commit()
        out.close()
        print(f"baked {baked} pre-rendered files ({skipped} already present) in {time.time()-t0:.0f}s")
        return

    # live path: fetch CGM bytes from imagerepository and render with the converter
    cmd = _env("VIDA_CGM_CMD")
    if not cmd:
        if shutil.which("java"):
            print("VIDA_CGM_CMD is not set. Provide a jcgm invocation, e.g. (jcgm-core -> PNG):\n"
                  "  VIDA_CGM_CMD='java -cp jcgm-core.jar:. CgmToPng {in} {out}'\n"
                  "(build recipe in docs/images-plan.md). Nothing baked.")
        else:
            print("No CGM converter available: Java is not installed and VIDA_CGM_CMD is unset.\n"
                  "Install a JDK + build jcgm (docs/images-plan.md), then set VIDA_CGM_CMD, or\n"
                  "render the CGMs elsewhere and bake them with --render-dir. Nothing baked — CGMs\n"
                  "remain a labelled note in the UI (this is the documented, non-fatal default).")
        out.close()
        return

    user, pw = _env("VIDA_SQL_USER"), _env("VIDA_SQL_PASSWORD")
    if not user or not pw:
        sys.exit("VIDA_SQL_USER / VIDA_SQL_PASSWORD missing (.env) — needed for CGM bytes")
    import pytds
    con = pytds.connect(server="127.0.0.1", port=1433, database="imagerepository",
                        user=user, password=pw, autocommit=True, login_timeout=5, timeout=60)
    cur = con.cursor()
    missing = failed = 0
    for cgm in cgms:
        if _already_baked(cgm, have) and not args.force:
            skipped += 1
            continue
        cur.execute("SELECT imageData FROM LocalizedGraphics WHERE path = %s", (cgm,))
        row = cur.fetchone()
        if not (row and row[0]):
            missing += 1
            continue
        data = convert(bytes(row[0]), cmd)
        if data:
            ext, ct = _sniff(data)
            sp = rendered_path_for(cgm, ext)
            out.execute("INSERT OR REPLACE INTO images VALUES (?,?,?)", (sp, ct, data))
            have.add(sp)
            baked += 1
            print(f"  baked {sp} ({ct}, {len(data)} bytes)", flush=True)
        else:
            failed += 1
    out.commit()
    con.close()
    out.close()
    print(f"done in {time.time()-t0:.0f}s | baked {baked} rendered, {skipped} already present, "
          f"{missing} CGM bytes not in repo, {failed} conversions failed", flush=True)


if __name__ == "__main__":
    main()
