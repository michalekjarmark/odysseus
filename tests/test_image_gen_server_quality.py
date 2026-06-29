"""Regression: the agent's generate_image MCP server must send `quality` (and an
unclamped size) for LOCAL diffusion, not only for GPT image models.

Before the fix, quality was added to the payload only when is_gpt_image, so every
agent-driven SDXL render fell back to the diffusion server's 8-step default
(fast but soft/garbled) while the direct do_generate_image path sent quality and
got the configured high (35) step count. Size was also wrongly clamped to the
DALL-E 3 menu, forcing local diffusion back to 1024x1024.
"""

import asyncio
import json

import src.agent_tools  # noqa: F401 — import first to avoid a circular import
import mcp_servers.image_gen_server as igs


class _FakeResp:
    status_code = 200

    def __init__(self):
        # A tiny 1x1 PNG so the b64 decode + file write path succeeds.
        self._b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

    def json(self):
        return {"data": [{"b64_json": self._b64}]}


class _FakeAsyncClient:
    captured = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None, headers=None):
        _FakeAsyncClient.captured = {"url": url, "json": json, "headers": headers}
        return _FakeResp()


def _run(arguments, monkeypatch, *, image_quality="high", model_id="stable-diffusion-xl-base-1.0"):
    import src.settings as settings
    import src.ai_interaction as ai

    monkeypatch.setattr(settings, "get_setting",
                        lambda key, default=None: {
                            "image_gen_enabled": True,
                            "image_model": model_id,
                            "image_quality": image_quality,
                            "app_public_url": "",
                        }.get(key, default))
    monkeypatch.setattr(settings, "load_settings",
                        lambda: {"image_model": model_id, "image_quality": image_quality})
    monkeypatch.setattr(ai, "_resolve_model",
                        lambda spec, owner=None: ("http://127.0.0.1:8100/v1", model_id, {}))
    # httpx is imported inside call_tool; patch the module-level httpx it imports.
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    asyncio.run(igs.call_tool("generate_image", arguments))
    return _FakeAsyncClient.captured["json"]


def test_local_diffusion_payload_includes_quality(monkeypatch):
    payload = _run({"prompt": "a fox"}, monkeypatch, image_quality="high")
    # The admin "high" setting must reach the diffusion server (which maps it to
    # the high step count), so agent renders match the direct path.
    assert payload["quality"] == "high"
    assert payload["prompt"] == "a fox"


def test_local_diffusion_size_not_clamped_to_dalle_menu(monkeypatch):
    payload = _run({"prompt": "a fox", "size": "1024x1536"}, monkeypatch)
    assert payload["size"] == "1024x1536"


def test_gpt_image_still_gets_quality(monkeypatch):
    payload = _run({"prompt": "a fox"}, monkeypatch, model_id="gpt-image-1")
    assert payload["quality"] == "high"
