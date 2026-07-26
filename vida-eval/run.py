"""vida-eval: regression set for the VIDA S60R assistant.

Runs the question set in cases.json against the live stack and grades answers
mechanically (must_mention / must_not_say / tool_must_not_contain). Multi-turn
cases replay follow-up history the way the Open WebUI chat does: only the final
assistant TEXT of earlier turns is kept (tool dumps are stripped).

Backends:
  native (default)  LM Studio /v1/chat/completions + vida-kb MCP directly.
                    This mirrors what the Open WebUI UI does in native
                    function-calling mode and needs no Open WebUI token.
  owui              POST $OWUI_URL/api/chat/completions with $OWUI_TOKEN.
                    NOTE: with the bare API and a native-function-calling
                    preset, Open WebUI hands tool calls back to the client
                    instead of executing them; use a default-function-calling
                    preset for this backend, or prefer the native backend.

Usage:
  uv run python vida-eval/run.py                       # full set
  uv run python vida-eval/run.py --case contamination-unrelated
  uv run python vida-eval/run.py --system-file /tmp/old-prompt.md
  uv run python vida-eval/run.py --json-out vida-eval/results/run.json
  OWUI_TOKEN=... uv run python vida-eval/run.py --backend owui --model vida-s60r

Exit code: 0 = all graded checks passed, 1 = any failure.
"""
import argparse
import asyncio
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LMS = os.environ.get("LMS_URL", "http://127.0.0.1:1234") + "/v1/chat/completions"
MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8765/mcp")
OWUI_URL = os.environ.get("OWUI_URL", "http://127.0.0.1:8080")


def load_system(path: str | None) -> str:
    p = Path(path) if path else ROOT / "docs" / "system-prompt.md"
    text = p.read_text()
    # the prompt file has a "# heading" + blank line before the prompt body
    return text.split("\n", 2)[2] if text.startswith("#") else text


def llm(messages, tools, model):
    body = {"model": model, "messages": messages, "stream": False, "temperature": 0.1}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        LMS, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]


async def run_case_native(case, system, model, max_rounds):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    turn_results = []
    async with streamablehttp_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            listed = await s.list_tools()
            tools = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema}}
                for t in listed.tools
            ]
            history = [{"role": "system", "content": system}]
            for turn in case["turns"]:
                history.append({"role": "user", "content": turn["q"]})
                messages = list(history)  # turn-local copy holds tool traffic
                calls, answer = [], "(no answer: tool-loop limit reached)"
                for _ in range(max_rounds):
                    msg = llm(messages, tools, model)
                    if not msg.get("tool_calls"):
                        answer = (msg.get("content") or "").strip()
                        break
                    messages.append(msg)
                    for tc in msg["tool_calls"]:
                        name = tc["function"]["name"]
                        args = tc["function"]["arguments"] or "{}"
                        calls.append({"tool": name, "args": args})
                        res = await s.call_tool(name, json.loads(args))
                        content = "\n".join(c.text for c in res.content if getattr(c, "text", None))
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content[:30000]})
                # follow-up history mirrors the UI: final text only, no tool dumps
                history.append({"role": "assistant", "content": answer})
                turn_results.append({"q": turn["q"], "answer": answer, "tool_calls": calls})
    return turn_results


def run_case_owui(case, system, model):
    token = os.environ.get("OWUI_TOKEN")
    if not token:
        sys.exit("owui backend needs OWUI_TOKEN (ask the Open WebUI admin for an API key)")
    turn_results = []
    history = []  # owui injects the preset's own system prompt; --system-file is ignored here
    for turn in case["turns"]:
        history.append({"role": "user", "content": turn["q"]})
        body = {"model": model, "messages": history, "stream": False}
        req = urllib.request.Request(
            f"{OWUI_URL}/api/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        )
        msg = json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]
        answer = (msg.get("content") or "").strip()
        history.append({"role": "assistant", "content": answer})
        # bare-API limitation: server-side tool calls are not visible to us here
        turn_results.append({"q": turn["q"], "answer": answer, "tool_calls": None})
    return turn_results


def grade(case, turns):
    failures = []
    for i, (spec, got) in enumerate(zip(case["turns"], turns), 1):
        low = got["answer"].lower()
        for m in spec.get("must_mention", []):
            if m.lower() not in low:
                failures.append(f"turn {i}: missing must_mention {m!r}")
        for m in spec.get("must_not_say", []):
            if m.lower() in low:
                failures.append(f"turn {i}: said banned {m!r}")
        banned_tools = spec.get("tool_must_not_contain", [])
        must_use = spec.get("tool_must_use", [])
        if (banned_tools or must_use) and got["tool_calls"] is None:
            failures.append(f"turn {i}: tool calls not observable on this backend; cannot check tool expectations")
        for m in banned_tools:
            for c in got["tool_calls"] or []:
                if m.lower() in c["args"].lower():
                    failures.append(f"turn {i}: tool call {c['tool']}({c['args'][:120]}) contains banned {m!r}")
        for name in must_use:
            if not any(c["tool"] == name for c in got["tool_calls"] or []):
                used = [c["tool"] for c in got["tool_calls"] or []]
                failures.append(f"turn {i}: expected tool {name!r} not used (used: {used})")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(HERE / "cases.json"))
    ap.add_argument("--case", action="append", help="run only these case id(s); repeatable")
    ap.add_argument("--backend", choices=["native", "owui"], default="native")
    ap.add_argument("--model", default=None,
                    help="native: LM Studio model id (default qwen/qwen3.6-35b-a3b); owui: preset id (default vida-s60r)")
    ap.add_argument("--system-file", default=None, help="override system prompt (native backend only)")
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    model = args.model or ("qwen/qwen3.6-35b-a3b" if args.backend == "native" else "vida-s60r")
    system = load_system(args.system_file)
    spec = json.loads(Path(args.cases).read_text())
    cases = [c for c in spec["cases"] if not args.case or c["id"] in args.case]
    if args.case and len(cases) != len(set(args.case)):
        sys.exit(f"unknown case id(s): {set(args.case) - {c['id'] for c in cases}}")

    results, any_fail = [], False
    for case in cases:
        print(f"=== {case['id']} ({len(case['turns'])} turn{'s' if len(case['turns']) > 1 else ''})", flush=True)
        if args.backend == "native":
            turns = asyncio.run(run_case_native(case, system, model, args.max_rounds))
        else:
            turns = run_case_owui(case, system, model)
        failures = grade(case, turns)
        any_fail |= bool(failures)
        for t in turns:
            for c in t["tool_calls"] or []:
                print(f"    -> {c['tool']}({c['args'][:120]})", flush=True)
        status = "PASS" if not failures else "FAIL: " + "; ".join(failures)
        print(f"  {status}\n", flush=True)
        results.append({"id": case["id"], "pass": not failures, "failures": failures, "turns": turns})

    passed = sum(1 for r in results if r["pass"])
    print(f"{passed}/{len(results)} cases passed")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"backend": args.backend, "model": model, "system_file": args.system_file, "results": results},
            indent=2))
        print(f"results written to {out}")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
