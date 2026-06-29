"""Regression: recover a tool call a weak local model emits as a JSON object.

gemma4:e4b (and similar small models) sometimes can't emit the native/fenced
tool format and instead PRINT the call as a JSON object in prose or a ```json
fence, e.g. {"tool_name": "generate_image", "params": {...}}. No fenced-tool /
native pattern matched it, so the call was silently dropped — the model "showed"
the tool instead of running it. parse_tool_blocks now recovers it (fallback only,
and only for non-native models where skip_fenced is False).
"""

import src.agent_tools  # noqa: F401 — import first to avoid a circular import
from src.tool_parsing import parse_tool_blocks, strip_tool_blocks
from src.tool_execution import _parse_generate_image


_FENCED = (
    "I will execute that request now.\n\n"
    "```json\n"
    "{\n"
    '  "tool_name": "generate_image",\n'
    '  "params": {"prompt": "hyperrealistic portrait, 85mm lens", "size": "1024x1536"}\n'
    "}\n"
    "```"
)


def test_recovers_fenced_json_tool_call_with_params():
    blocks = parse_tool_blocks(_FENCED, skip_fenced=False)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "generate_image"
    args = _parse_generate_image(blocks[0].content)
    assert args["prompt"].startswith("hyperrealistic portrait")
    assert args["size"] == "1024x1536"


def test_strip_removes_the_printed_json_and_empty_fence():
    assert strip_tool_blocks(_FENCED, skip_fenced=False) == "I will execute that request now."


def test_recovers_flat_name_plus_args():
    blocks = parse_tool_blocks('{"name": "generate_image", "prompt": "a red fox"}', skip_fenced=False)
    assert len(blocks) == 1 and blocks[0].tool_type == "generate_image"
    assert _parse_generate_image(blocks[0].content)["prompt"] == "a red fox"


def test_recovers_nested_openai_function_shape():
    raw = '{"function": {"name": "generate_image", "arguments": {"prompt": "a cat"}}}'
    blocks = parse_tool_blocks(raw, skip_fenced=False)
    assert len(blocks) == 1 and blocks[0].tool_type == "generate_image"
    assert _parse_generate_image(blocks[0].content)["prompt"] == "a cat"


def test_recovers_tool_name_via_alias_or_fuzzy():
    # Skill-heading style name should resolve through function_call_to_tool_block.
    raw = '{"tool_name": "generate_images-with-the-image-tool", "params": {"prompt": "x"}}'
    blocks = parse_tool_blocks(raw, skip_fenced=False)
    assert len(blocks) == 1 and blocks[0].tool_type == "generate_image"


def test_casual_json_is_not_a_tool_call():
    assert parse_tool_blocks('Here is data: {"name": "John", "age": 30}', skip_fenced=False) == []
    assert parse_tool_blocks('{"temperature": 0.7, "top_p": 0.9}', skip_fenced=False) == []


def test_native_models_do_not_recover_printed_json():
    # skip_fenced=True (native function-calling model): a bare JSON object is
    # illustrative display text, never an action.
    assert parse_tool_blocks(_FENCED, skip_fenced=True) == []


def test_real_tool_blocks_take_precedence():
    # A genuine fenced generate_image block must win; the JSON fallback only runs
    # when nothing else matched.
    text = "```generate_image\na sunset over mountains\n```"
    blocks = parse_tool_blocks(text, skip_fenced=False)
    assert len(blocks) == 1 and blocks[0].tool_type == "generate_image"
    assert blocks[0].content.strip() == "a sunset over mountains"
