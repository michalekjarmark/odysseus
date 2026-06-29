"""A document auto-created from a code block must END the turn, not loop.

When a model dumps a large code block in chat with no real tool call, the agent
auto-creates a document from it. That document is the model's FINAL answer — there
is nothing for the model to react to. Looping (feeding it back for another round)
makes weak local models (e.g. qwen3.5:9b) re-emit the whole document over and over
until the context runs out. The turn must terminate after the one round.
"""

import asyncio
import json

import src.agent_loop as agent_loop


def _collect(gen):
    async def _run():
        return [chunk async for chunk in gen]
    return asyncio.run(_run())


def _events(chunks):
    out = []
    for chunk in chunks:
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            out.append(json.loads(chunk[6:]))
    return out


def _big_html():
    body = "\n".join(f"  <div>line {i}</div>" for i in range(40))  # >30 lines
    return "<!DOCTYPE html>\n<html>\n<body>\n" + body + "\n</body>\n</html>"


def test_auto_created_document_is_terminal(monkeypatch):
    monkeypatch.setattr(agent_loop, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(agent_loop, "estimate_tokens", lambda *a, **k: 10, raising=False)

    calls = {"n": 0}
    text = "Here is your page:\n\n```html\n" + _big_html() + "\n```\n"

    async def fake_stream(_candidates, messages, **kwargs):
        calls["n"] += 1
        yield f'data: {json.dumps({"delta": text})}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        return ("create_document", {
            "action": "create", "doc_id": "d1", "title": "Code (html)",
            "content": block.content, "version": 1, "language": "html",
            "output": "Document created", "exit_code": None,
        })

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    chunks = _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o",
        [{"role": "user", "content": "make me an html quiz page"}],
        relevant_tools={"create_document"},
        session_id="s1",
        _is_teacher_run=True,
    ))
    events = _events(chunks)

    # The model was called exactly once — no second round re-emitting the doc.
    assert calls["n"] == 1, f"expected a single round, got {calls['n']}"
    # The document was still created (the answer isn't lost).
    metrics = next(e["data"] for e in events if e.get("type") == "metrics")
    tools = [te["tool"] for te in metrics.get("tool_events", [])]
    assert tools.count("create_document") == 1


def test_real_tool_call_plus_codeblock_still_loops(monkeypatch):
    """Guard: when the model made a REAL tool call this round, a code block in the
    same text must NOT terminate — the agent still needs to react to the tool. Uses
    a NATIVE tool call (parsed regardless of model type) alongside the big block."""
    monkeypatch.setattr(agent_loop, "get_setting", lambda key, default=None: default, raising=False)
    monkeypatch.setattr(agent_loop, "get_mcp_manager", lambda: None, raising=False)
    monkeypatch.setattr(agent_loop, "estimate_tokens", lambda *a, **k: 10, raising=False)

    calls = {"n": 0}
    text = "Reading first, here's a draft:\n\n```html\n" + _big_html() + "\n```\n"

    async def fake_stream(_candidates, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            call = {"name": "bash", "arguments": json.dumps({"command": "ls -la"})}
            yield f'data: {json.dumps({"type": "tool_calls", "calls": [call]})}\n\n'
            yield f'data: {json.dumps({"delta": text})}\n\n'
        else:
            # Second round: a plain final answer, no tool → terminate.
            yield f'data: {json.dumps({"delta": "All done."})}\n\n'
        yield "data: [DONE]\n\n"

    async def fake_execute(block, *args, **kwargs):
        if block.tool_type == "bash":
            return ("bash", {"output": "total 0", "exit_code": 0})
        return ("create_document", {
            "action": "create", "doc_id": "d1", "title": "Code (html)",
            "content": block.content, "version": 1, "language": "html",
            "output": "Document created", "exit_code": None,
        })

    monkeypatch.setattr(agent_loop, "stream_llm_with_fallback", fake_stream, raising=False)
    monkeypatch.setattr(agent_loop, "execute_tool_block", fake_execute, raising=False)

    _collect(agent_loop.stream_agent_loop(
        "https://api.openai.com/v1", "gpt-4o",
        [{"role": "user", "content": "list the files and draft a page"}],
        relevant_tools={"bash", "create_document"},
        session_id="s1",
        _is_teacher_run=True,
    ))
    # A real (native) bash call must feed back and run a second round, even though a
    # big code block was also present in the same text.
    assert calls["n"] == 2, f"expected a real tool call to loop, got {calls['n']}"
