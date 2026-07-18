"""
title: VIDA Citation Links
author: vida-bot
version: 0.1.0
required_open_webui_version: 0.5.0
description: >
  Rewrites bare VIDA citation tokens (e.g. "VCC-461742-2") in the assistant's answer body
  into clickable Markdown links to the VIDA document server's /doc/<ref> route, so a user
  can tap a citation and read the exact source document that grounded the answer. Pairs with
  the /doc/ route added to src/server.py (served on the same private-network-bound host:port as the
  figure /img/ server). No model change, no retraining — pure post-processing of the answer.

INSTALL (Open WebUI):
  Admin Panel -> Functions -> + (New Function) -> paste this file -> Save.
  Enable it globally, or per-model on the VIDA model. Set the Valve `doc_base_url` to your
  VIDA doc server (default = the private host:port the figure server already uses).
  The model must already cite sources as VCC tokens (the grounding contract does this).
"""

import re
from pydantic import BaseModel, Field

# Skip a token already inside a URL ("/VCC-…") or already link text ("[VCC-…]") so the
# rewrite is idempotent across repeated outlet calls and never double-links.
_VCC_RE = re.compile(r"(?<![/\[])\bVCC-\d{6}(?:-\d+)?\b")


class Filter:
    class Valves(BaseModel):
        enabled: bool = Field(default=True, description="Master on/off for citation linking.")
        doc_base_url: str = Field(
            default="http://localhost:8766",
            description="Base URL of the VIDA /doc server (the private host:port from "
                        "src/server.py IMAGE_HOST/IMAGE_PORT). Links point at <base>/doc/<VCC-ref>.",
        )

    def __init__(self):
        self.valves = self.Valves()

    def outlet(self, body: dict, __user__: dict | None = None) -> dict:
        """Post-process: rewrite VCC tokens in assistant messages into clickable links."""
        if not self.valves.enabled:
            return body
        base = (self.valves.doc_base_url or "").rstrip("/")
        if not base:
            return body

        def _link(m: "re.Match") -> str:
            tok = m.group(0)
            return f"[{tok}]({base}/doc/{tok})"

        for msg in body.get("messages", []):
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                msg["content"] = _VCC_RE.sub(_link, msg["content"])
        return body
