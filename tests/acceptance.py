"""Phase-1 acceptance bench: native tool-call loop (LM Studio + vida-kb MCP).

Replicates what Open WebUI's UI does in native function-calling mode: the model
gets the MCP tools, iterates search -> read -> answer, and must cite sources.

  uv run python tests/acceptance.py            # run all questions
  uv run python tests/acceptance.py 2          # run one question by index
"""
import asyncio
import json
import sys
from pathlib import Path

import urllib.request

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

LMS = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "qwen/qwen3.6-35b-a3b"
MCP_URL = "http://127.0.0.1:8765/mcp"
SYSTEM = (Path(__file__).resolve().parent.parent / "docs" / "system-prompt.md").read_text().split("\n", 2)[2]

QUESTIONS = [
    "What is the tightening torque for the M6 intake manifold bolts on the B5254T4? Include the procedure doc if there is one.",
    "What is the install procedure for the exhaust manifold studs?",
    "What are the bolt lengths for the charge air pipe / charge air cooler installation?",
    "Tell me about CEM pin A:6 — what does the wiring documentation say about it?",
    "What is the recommended tire pressure placard location for a 2019 Tesla Model 3?",
]


def llm(messages, tools):
    body = {"model": MODEL, "messages": messages, "tools": tools, "stream": False, "temperature": 0.1}
    req = urllib.request.Request(LMS, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]


async def run(question: str) -> tuple[str, list[str]]:
    async with streamablehttp_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            listed = await s.list_tools()
            tools = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema}}
                for t in listed.tools
            ]
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
            calls_made = []
            for _ in range(8):
                msg = llm(messages, tools)
                if not msg.get("tool_calls"):
                    return (msg.get("content") or "").strip(), calls_made
                messages.append(msg)
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    args = json.loads(tc["function"]["arguments"] or "{}")
                    calls_made.append(f"{name}({json.dumps(args)})")
                    res = await s.call_tool(name, args)
                    content = "\n".join(c.text for c in res.content if getattr(c, "text", None))
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content[:30000]})
            return "(loop limit reached)", calls_made


async def main():
    idxs = [int(sys.argv[1])] if len(sys.argv) > 1 else range(len(QUESTIONS))
    for i in idxs:
        q = QUESTIONS[i]
        print("=" * 78)
        print(f"Q{i}: {q}\n")
        answer, calls = await run(q)
        for c in calls:
            print(f"  -> {c[:150]}")
        print()
        print(answer[:2400])
        print()


if __name__ == "__main__":
    asyncio.run(main())
