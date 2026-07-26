# Pinouts

These CSVs are **not** vendored here. They live in their own repo so there is one
source of truth:

    https://github.com/jasongainor/volvo-p2-pinout

## Required before `make ingest`

`src/build_db.py` opens three files by name and loads them into the `pinouts` table,
which backs the `lookup_pin` tool. They are **not optional** — ingest raises
`FileNotFoundError` if they are absent:

    volvo_s60r_2005_ALL_PINOUT_verified.csv
    volvo_s60r_2005_ECM_pinout_verified.csv
    volvo_s60r_2005_components.csv

Fetch them into this directory:

    git clone --depth 1 https://github.com/jasongainor/volvo-p2-pinout /tmp/p2-pinout
    cp /tmp/p2-pinout/*.csv data/pinouts/

Or add that repo as a submodule if you want to pin a revision.
