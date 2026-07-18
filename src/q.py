"""Ad-hoc query CLI against the attached VIDA databases.

Usage:
  uv run python src/q.py <database> "<sql>"            # print rows, tab-separated
  uv run python src/q.py <database> "<sql>" --blob FILE  # write first column of first row (bytes) to FILE

Binary values are shown as <bytes:N:hexprefix> unless --blob is used.
"""
import sys

import pytds

from dbconfig import conn_params  # SQL creds from .env (see .env.example) — never hardcoded

CONN = conn_params()


def fmt(v):
    if isinstance(v, (bytes, bytearray)):
        return f"<bytes:{len(v)}:{bytes(v[:8]).hex()}>"
    if isinstance(v, str) and len(v) > 400:
        return v[:400] + f"...<+{len(v)-400} chars>"
    return v


def main():
    db, sql = sys.argv[1], sys.argv[2]
    blob_out = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--blob" else None
    with pytds.connect(database=db, **CONN) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        if cur.description is None:
            print("(no result set)")
            return
        if blob_out:
            row = cur.fetchone()
            data = bytes(row[0])
            with open(blob_out, "wb") as f:
                f.write(data)
            print(f"wrote {len(data)} bytes to {blob_out}")
            return
        print("\t".join(d[0] for d in cur.description))
        for row in cur.fetchall():
            print("\t".join(str(fmt(v)) for v in row))


if __name__ == "__main__":
    main()
