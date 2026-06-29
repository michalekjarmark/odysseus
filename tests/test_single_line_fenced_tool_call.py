"""Regression: a single-line fenced tool call ```tag arg``` must be recovered.

Some weak local GGUF finetunes emit the whole call on ONE line with the closing
fence on the same line, e.g. ```read_file "C:\\path\\file.txt"``` — instead of the
standard multi-line body. The Pattern-1 regex required ``\\s*\\n`` right after the
tag, so the single-line form matched NOTHING — and not just for read_file: every
tool tag (bash, ls, glob, web_search, ...) shared that regex, so all of them were
silently dropped. The tag is now followed by EITHER a newline OR inline whitespace.
"""

import src.agent_tools  # noqa: F401  (break agent_tools<->tool_parsing import cycle)
from src.tool_parsing import parse_tool_blocks, strip_tool_blocks


def test_single_line_read_file_double_quotes():
    text = 'Let me read it:\n\n```read_file "D:\\data\\uploads\\file.txt"```'
    blocks = parse_tool_blocks(text, skip_fenced=False)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "read_file"
    # The wrapping quotes are stripped so the path isn't taken literally.
    assert blocks[0].content == "D:\\data\\uploads\\file.txt"


def test_single_line_read_file_single_quotes():
    blocks = parse_tool_blocks("```read_file '/etc/hosts'```", skip_fenced=False)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "read_file"
    assert blocks[0].content == "/etc/hosts"


def test_single_line_recovers_every_shared_tag():
    # The same regex powered all tags, so all were broken in the inline form.
    cases = {
        "```ls .```": ("ls", "."),
        "```glob **/*.py```": ("glob", "**/*.py"),
        "```bash ls -la```": ("bash", "ls -la"),
        "```python print(1)```": ("python", "print(1)"),
        "```web_search latest AI news```": ("web_search", "latest AI news"),
    }
    for text, (tag, content) in cases.items():
        blocks = parse_tool_blocks(text, skip_fenced=False)
        assert len(blocks) == 1, text
        assert blocks[0].tool_type == tag, text
        assert blocks[0].content == content, text


def test_shell_tags_keep_their_quotes():
    # bash/python are NOT unwrapped — quotes are meaningful in shell/code.
    blocks = parse_tool_blocks('```bash echo "hi there"```', skip_fenced=False)
    assert len(blocks) == 1 and blocks[0].tool_type == "bash"
    assert blocks[0].content == 'echo "hi there"'


def test_multiline_form_still_works():
    # No regression for the standard newline-after-tag body.
    blocks = parse_tool_blocks("```read_file\nD:\\path\\file.txt\n```", skip_fenced=False)
    assert len(blocks) == 1 and blocks[0].tool_type == "read_file"
    assert blocks[0].content == "D:\\path\\file.txt"


def test_quoted_path_in_multiline_read_file_body_is_unwrapped():
    blocks = parse_tool_blocks('```read_file\n"D:\\path\\file.txt"\n```', skip_fenced=False)
    assert len(blocks) == 1 and blocks[0].tool_type == "read_file"
    assert blocks[0].content == "D:\\path\\file.txt"


def test_empty_no_arg_fence_is_not_a_call():
    assert parse_tool_blocks("```read_file```", skip_fenced=False) == []
    assert parse_tool_blocks("```bash```", skip_fenced=False) == []


def test_native_model_still_ignores_single_line_fence():
    # skip_fenced=True (native function-calling model): fenced text is display only.
    assert parse_tool_blocks('```read_file "/etc/hosts"```', skip_fenced=True) == []


def test_strip_removes_single_line_fenced_call():
    text = 'Reading now:\n\n```read_file "/etc/hosts"```'
    cleaned = strip_tool_blocks(text, skip_fenced=False)
    assert "```" not in cleaned
    assert "read_file" not in cleaned
    assert "Reading now:" in cleaned
