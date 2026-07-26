"""dataset/torque.py — the torque PARSER (DECISIONS D6).

`LIKE '%Nm%'` is polluted (7,224 docs, most without a real value); the signal is
~2,198 docs with a value and ~125 clean 'Tightening torque' spec tables, some
malformed ('| Oil drain plug40 Nm |'). This module turns a document's text_md into
[(component, value_nm, unit)] triples and, crucially, reports MEASURED precision —
we never claim "100% deterministic torque".

Two extraction paths:
  1. markdown spec-table rows: '| Oil drain plug | 40 Nm |' AND the malformed
     no-separator form '| Oil drain plug40 Nm |' (trailing-number rescue).
  2. inline prose: 'Tighten ... to 40 Nm', 'Tightening torque: 15 Nm'.
Cross-reference stubs ('For tightening torques, see [VCC-142128-1]') carry NO value
and are discarded.

Pure stdlib (re).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# A torque magnitude + unit, e.g. "40 Nm", "15 N·m", "8.5 Nm". Capture the number.
_VALUE = r"(\d+(?:[.,]\d+)?)\s*(N[\s·.\-]?m|Nm)"
_VALUE_RE = re.compile(_VALUE, re.I)

# Inline prose: "...tighten ... to 40 Nm" / "Tightening torque: 15 Nm" / "Torque: 22 Nm"
_INLINE_RE = re.compile(
    r"(?P<comp>[A-Za-z][\w \-/()]{2,60}?)\s*"
    r"(?:tightening torque|torque)\s*[:\-]?\s*"
    r"(?P<val>\d+(?:[.,]\d+)?)\s*(?P<unit>N[\s·.\-]?m|Nm)",
    re.I,
)
_TIGHTEN_RE = re.compile(
    r"(?:tighten(?:ing)?|torque)\b[^.\n]{0,60}?\bto\s+"
    r"(?P<val>\d+(?:[.,]\d+)?)\s*(?P<unit>N[\s·.\-]?m|Nm)",
    re.I,
)

# A cross-reference stub with no number: discard.
_STUB_RE = re.compile(r"\bsee\b.*\bVCC-\d", re.I)


@dataclass
class Torque:
    component: str
    value_nm: float
    unit: str = "Nm"

    def as_tuple(self) -> tuple[str, float, str]:
        return (self.component, self.value_nm, self.unit)


def _num(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def _clean_component(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    # drop markdown emphasis (literal, so legitimate underscores in identifiers survive)
    s = s.replace("**", "").replace("__", "")
    # drop a trailing "(NOTE! ...)" clause that VIDA spec rows append to the component
    s = re.sub(r"\s*\(NOTE!.*$", "", s, flags=re.I)
    s = s.strip(" -|:*•\t")
    # strip a leading list/callout marker like "1." or "- "
    s = re.sub(r"^\d+[.)]\s*", "", s)
    return s.strip()


def _parse_table_row(row: str) -> Optional[Torque]:
    """A markdown table row '| component | 40 Nm |' or the malformed
    '| component40 Nm |' (no separator). Returns None if no usable value."""
    if "Nm" not in row and "N" not in row:
        return None
    # Precision-first: a multi-stage / multi-value row (e.g. "Step 1: 20 Nm Step 2:
    # 130 Nm" or "| 20 Nm | 60 Nm | 90° |") must NOT be collapsed to a single
    # magnitude — that silently emits the wrong torque. Refuse it instead. (~5% of
    # real spec-table rows are multi-value; see docs/AUDIT-2026-06-20.md.)
    _mags = {round(_n, 3) for _n in (_num(g[0]) for g in _VALUE_RE.findall(row))
             if _n is not None}
    if len(_mags) >= 2:
        return None
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    # Well-formed: some cell is the component, another holds the value.
    val: Optional[float] = None
    unit = "Nm"
    comp_cells: list[str] = []
    for c in cells:
        m = _VALUE_RE.search(c)
        if m and _num(m.group(1)) is not None and not c.replace(" ", "").lower().startswith("n"):
            # a cell that is essentially just the value
            stripped = _VALUE_RE.sub("", c).strip()
            if len(stripped) <= 2:  # cell was basically just "40 Nm"
                val = _num(m.group(1))
                unit = re.sub(r"[\s·.\-]", "", m.group(2)).title()
                continue
        comp_cells.append(c)
    component = _clean_component(" ".join(x for x in comp_cells if x))
    # Malformed no-separator rescue: 'Oil drain plug40 Nm' -> component + trailing value.
    if val is None:
        m = _VALUE_RE.search(component)
        if m:
            val = _num(m.group(1))
            unit = re.sub(r"[\s·.\-]", "", m.group(2)).title()
            component = _clean_component(component[: m.start()])
    if val is None or not component or len(component) < 3:
        return None
    return Torque(component=component, value_nm=val, unit=unit or "Nm")


def parse_torque(text: str) -> list[Torque]:
    """All (component, value_nm, unit) torques parseable from `text_md`. Spec-table
    rows first (highest precision), then inline prose. Cross-ref stubs discarded.
    Deduped on (component_lower, value)."""
    if not text:
        return []
    out: list[Torque] = []
    seen: set[tuple[str, float]] = set()

    def add(t: Optional[Torque]) -> None:
        if t is None:
            return
        key = (t.component.lower(), t.value_nm)
        if key not in seen:
            seen.add(key)
            out.append(t)

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _STUB_RE.search(line) and not _VALUE_RE.search(line):
            continue  # "see VCC-..." with no value
        if line.startswith("|") and line.count("|") >= 2:
            add(_parse_table_row(line))
            continue
        m = _INLINE_RE.search(line)
        if m:
            comp = _clean_component(m.group("comp"))
            val = _num(m.group("val"))
            if comp and val is not None and len(comp) >= 3:
                add(Torque(comp, val, re.sub(r"[\s·.\-]", "", m.group("unit")).title()))
            continue
        m = _TIGHTEN_RE.search(line)
        if m:
            val = _num(m.group("val"))
            # component = the text before the tighten verb on this line, if meaningful
            comp = _clean_component(line[: m.start()]) or "the fastener"
            if val is not None:
                add(Torque(comp[:60], val, re.sub(r"[\s·.\-]", "", m.group("unit")).title()))
    return out


def first_torque(text: str) -> Optional[Torque]:
    vals = parse_torque(text)
    return vals[0] if vals else None


def measure_precision(samples: list[tuple[str, Optional[float]]]) -> dict:
    """Hand-validation harness (D6: report a number, don't claim 100%).

    `samples` = [(text_md, expected_first_value_nm or None)]. Returns
    {n, parsed, correct, precision, recall} where 'correct' counts samples whose
    first parsed value equals the hand-labeled expected value (None == 'no torque
    here' and we expect to parse nothing)."""
    n = len(samples)
    parsed = 0           # samples we extracted a value from
    true_pos = 0         # parsed AND matches the hand-labeled expected value
    true_neg = 0         # expected no torque AND we parsed none
    expected_have = 0    # samples that genuinely contain a torque value
    for text, expected in samples:
        got = first_torque(text)
        if got is not None:
            parsed += 1
        if expected is None:
            if got is None:
                true_neg += 1
        else:
            expected_have += 1
            if got is not None and abs(got.value_nm - expected) <= 0.05:
                true_pos += 1
    return {
        "n": n,
        "parsed": parsed,
        "true_pos": true_pos,
        "true_neg": true_neg,
        # precision: of the values we extracted, how many were the right value
        "precision": round(true_pos / parsed, 4) if parsed else 0.0,
        # recall: of the values that exist, how many we extracted correctly
        "recall": round(true_pos / expected_have, 4) if expected_have else 0.0,
    }


if __name__ == "__main__":
    cases = [
        ("| Oil drain plug | 40 Nm |", 40.0),
        ("| Oil drain plug40 Nm |", 40.0),                 # malformed, no separator
        ("| Wheels110 Nm |", 110.0),
        ("| **Cylinder head** | 40 Nm |", 40.0),           # markdown emphasis stripped
        ("Tightening torque: 15 Nm", 15.0),
        ("Tighten the bolts to 22 Nm in two stages.", 22.0),
        ("Intake manifold bolts tightening torque 19 Nm", 19.0),
        ("For tightening torques, see [VCC-142128-1]", None),  # stub -> nothing
        ("This procedure has no torque at all.", None),
        ("Spark plug torque: 8.5 N·m", 8.5),
        # multi-stage rows must REFUSE rather than emit one (wrong) value:
        ("| Cylinder head | 20 Nm | 60 Nm | 90° |", None),
        ("| **Wheel to hub** | Step 1: 20 Nm Step 2: 130 Nm |", None),
    ]
    prec = measure_precision(cases)
    for text, exp in cases:
        got = first_torque(text)
        flag = "ok" if ((exp is None and got is None) or
                        (exp is not None and got and abs(got.value_nm - exp) <= 0.05)) else "MISS"
        print(f"  [{flag}] {text[:48]!r:50} -> {got.as_tuple() if got else None}")
    print("precision report:", prec)
    assert prec["true_pos"] + prec["true_neg"] >= 10, prec
    assert prec["precision"] <= 1.0 and prec["recall"] <= 1.0, prec
    # markdown / multi-stage regressions:
    assert _clean_component("**Cylinder head**") == "Cylinder head"
    assert _parse_table_row("| Cylinder head | 20 Nm | 60 Nm | 90° |") is None
    assert _parse_table_row("| Wheel to hub | Step 1: 20 Nm Step 2: 130 Nm |") is None
    assert first_torque("| **Cylinder head** | 40 Nm |").component == "Cylinder head"
    print("torque.py self-check OK")
