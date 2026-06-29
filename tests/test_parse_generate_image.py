"""Regression: generate_image must accept BOTH the native JSON tool-call args and
the fenced-block line format.

Native tool calls arrive as a json.dumps'd object
(`{"prompt": "...", "size": "1024x1536"}`). The line-based parser used to take
that whole JSON string as line 0 = the prompt, so SDXL got a prompt polluted with
JSON syntax and truncated by CLIP's 77-token limit (garbage images), and the
requested size was dropped (everything came out 1024x1024).
"""

import json

import src.agent_tools  # noqa: F401 — import first to avoid a circular import
from src.tool_execution import _parse_generate_image


def test_native_json_args_extract_clean_prompt_and_size():
    raw = json.dumps({
        "prompt": "photorealistic portrait of a human, highly detailed, sharp focus, "
                  "professional studio lighting, natural skin texture",
        "size": "1024x1536",
    })
    args = _parse_generate_image(raw)
    assert not args["prompt"].lstrip().startswith("{")
    assert args["prompt"].startswith("photorealistic portrait of a human")
    assert '"prompt"' not in args["prompt"]  # no JSON syntax leaked in
    assert args["size"] == "1024x1536"        # the requested size is honoured


def test_native_json_with_model_and_quality():
    raw = json.dumps({"prompt": "a red fox", "model": "sdxl", "size": "1024x1024", "quality": "high"})
    assert _parse_generate_image(raw) == {
        "prompt": "a red fox", "model": "sdxl", "size": "1024x1024", "quality": "high",
    }


def test_line_based_format_still_parses():
    content = "a red fox in snow\nstable-diffusion-xl-base-1.0\n1024x1024\nhigh"
    assert _parse_generate_image(content) == {
        "prompt": "a red fox in snow",
        "model": "stable-diffusion-xl-base-1.0",
        "size": "1024x1024",
        "quality": "high",
    }


def test_plain_single_line_prompt():
    assert _parse_generate_image("just a cat") == {"prompt": "just a cat"}


def test_brace_text_that_is_not_json_is_kept_as_prompt():
    # A prompt that legitimately contains braces but isn't a JSON object must not
    # be swallowed — it falls back to the line-based parse.
    assert _parse_generate_image("a neon sign that says {hello}")["prompt"] == "a neon sign that says {hello}"


def test_image_prompt_key_alias():
    raw = json.dumps({"image_prompt": "a misty forest at dawn"})
    assert _parse_generate_image(raw)["prompt"] == "a misty forest at dawn"
