"""Shared VIDA SQL Server connection config.

Credentials are read from the environment, falling back to the gitignored `.env` file —
they are NEVER hardcoded in source. Copy `.env.example` to `.env` and set your own values.
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str | None = None) -> str | None:
    if key in os.environ:
        return os.environ[key]
    envf = _ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return default


def conn_params() -> dict:
    """pytds connection kwargs for the VIDA SQL Server. Exits if creds are unset."""
    user, pw = _env("VIDA_SQL_USER"), _env("VIDA_SQL_PASSWORD")
    if not user or not pw:
        sys.exit("VIDA_SQL_USER / VIDA_SQL_PASSWORD not set — copy .env.example to .env and fill them in")
    return dict(server=_env("VIDA_SQL_SERVER", "127.0.0.1"),
                port=int(_env("VIDA_SQL_PORT", "1433")),
                user=user, password=pw, autocommit=True)
