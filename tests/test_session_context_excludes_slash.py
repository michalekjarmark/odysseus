"""Regression: slash-command / setup messages must not reach LLM context.

Slash replies (and the echoed `/setup ...` command) are persisted to history so
they render in the transcript, tagged ``metadata.source == "slash"``. They are
UI chatter the user never meant as conversation, so ``get_context_messages``
(the LLM-API view) must exclude them while the raw history keeps them for
display. See issue #2634.
"""

from core.models import Session, ChatMessage


def _session_with_slash():
    s = Session(id="s1", name="t", endpoint_url="http://x/v1", model="m")
    s.add_message(ChatMessage("user", "hi, give me a recipe"))
    s.add_message(ChatMessage("user", "/setup copilot", metadata={"source": "slash"}))
    s.add_message(ChatMessage("assistant", "Starting GitHub Copilot sign-in...", metadata={"source": "slash"}))
    s.add_message(ChatMessage("assistant", "Here is a recipe", metadata={"model": "m"}))
    return s


def test_context_excludes_slash_messages():
    ctx = _session_with_slash().get_context_messages()
    contents = [m["content"] for m in ctx]
    assert "hi, give me a recipe" in contents
    assert "Here is a recipe" in contents
    # Slash command + its status reply are filtered out of LLM context.
    assert "/setup copilot" not in contents
    assert all("sign-in" not in c for c in contents)
    assert len(ctx) == 2


def test_history_still_keeps_slash_messages_for_display():
    s = _session_with_slash()
    # Raw history (what the UI renders) is untouched.
    assert len(s.history) == 4
    assert any(m.content == "/setup copilot" for m in s.history)


def test_no_metadata_messages_are_kept():
    s = Session(id="s2", name="t", endpoint_url="http://x/v1", model="m")
    s.add_message(ChatMessage("user", "plain"))
    s.add_message(ChatMessage("assistant", "reply"))
    assert [m["content"] for m in s.get_context_messages()] == ["plain", "reply"]


# --- tool_events replayed into context so the agent remembers its own actions ---


def test_generated_image_prompt_is_replayed_into_context():
    """Regression: in agent mode the model's tool results live only in
    metadata.tool_events; on a follow-up ("make it colored") it lost all record of
    what it had generated. The image prompt must now be surfaced into the content
    the LLM reads."""
    s = Session(id="s3", name="t", endpoint_url="http://x/v1", model="m")
    s.add_message(ChatMessage("user", "generate a hyperrealistic person"))
    s.add_message(ChatMessage(
        "assistant", "Here it is. Want to tweak the style?",
        metadata={"model": "m", "tool_events": [{
            "tool": "generate_image", "command": "a person",
            "image_prompt": "hyperrealistic portrait of a person, black and white",
            "image_model": "sdxl", "image_url": "/img/1.png",
        }]},
    ))
    s.add_message(ChatMessage("user", "i want to have colors"))
    ctx = s.get_context_messages()
    assistant = ctx[1]["content"]
    assert "Here it is." in assistant  # original prose preserved
    assert "hyperrealistic portrait of a person, black and white" in assistant
    assert "generate_image" in assistant


def test_assistant_without_tool_events_is_unchanged():
    s = Session(id="s4", name="t", endpoint_url="http://x/v1", model="m")
    s.add_message(ChatMessage("assistant", "plain reply", metadata={"model": "m"}))
    assert s.get_context_messages()[0]["content"] == "plain reply"


def test_tool_only_turn_with_none_content_gets_the_note():
    s = Session(id="s5", name="t", endpoint_url="http://x/v1", model="m")
    s.add_message(ChatMessage(
        "assistant", None,
        metadata={"tool_events": [{"tool": "bash", "command": "ls /tmp"}]},
    ))
    content = s.get_context_messages()[0]["content"]
    assert isinstance(content, str) and "bash" in content


def test_non_image_tool_events_do_not_echo_output():
    """Coding/file sessions must not be bloated: non-image tools contribute only
    name + command, never the (potentially huge) tool output."""
    s = Session(id="s6", name="t", endpoint_url="http://x/v1", model="m")
    big = "X" * 5000
    s.add_message(ChatMessage(
        "assistant", "done",
        metadata={"tool_events": [{"tool": "read_file", "command": "notes.txt", "output": big}]},
    ))
    content = s.get_context_messages()[0]["content"]
    assert big not in content
    assert "read_file" in content
