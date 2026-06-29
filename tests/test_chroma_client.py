"""Regression tests for the ChromaDB singleton client (issue #326).

Covers the fast-fail preflight (so an unreachable ChromaDB doesn't block
startup for the full OS connection timeout) and the rule that a failed
connection must not poison the cached singleton.
"""
import socket
import time

import pytest

import src.chroma_client as cc


def _free_port() -> int:
    """Bind to port 0, grab the assigned port, release it — nothing listens."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _embedded_supported(tmp) -> bool:
    """True only if the installed chromadb can actually run an embedded store.

    The HTTP-only `chromadb-client` build exposes PersistentClient but raises
    'http-only client mode' on use, so we probe by really creating one.
    """
    try:
        import chromadb
    except Exception:
        return False
    make = getattr(chromadb, "PersistentClient", None)
    if make is None:
        return False
    try:
        make(path=str(tmp)).heartbeat()
        return True
    except Exception:
        return False


def test_port_open_false_for_closed_port_and_is_fast():
    port = _free_port()
    t0 = time.monotonic()
    assert cc._port_open("127.0.0.1", port, timeout=1.0) is False
    # The whole point: we fail fast, nowhere near the 30-60s OS timeout.
    assert time.monotonic() - t0 < 5.0


def test_port_open_true_for_listening_socket():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()
    try:
        assert cc._port_open(host, port, timeout=1.0) is True
    finally:
        srv.close()


def test_get_chroma_client_does_not_cache_when_unreachable(monkeypatch):
    pytest.importorskip("chromadb")
    cc.reset_client()
    # Explicit server env => HTTP mode (strict): a down server is a real error,
    # never silently masked by the embedded fallback.
    monkeypatch.setenv("CHROMADB_HOST", "127.0.0.1")
    monkeypatch.setenv("CHROMADB_PORT", str(_free_port()))
    with pytest.raises(RuntimeError):
        cc.get_chroma_client()
    # A failed connection must leave the singleton unset so a later call
    # (once ChromaDB is up) can succeed.
    assert cc._client is None


def test_auto_mode_uses_embedded_when_no_server_configured(monkeypatch, tmp_path):
    pytest.importorskip("chromadb")
    if not _embedded_supported(tmp_path / "_probe"):
        pytest.skip("HTTP-only chromadb-client: embedded PersistentClient not usable")
    cc.reset_client()
    # No CHROMADB_HOST/PORT/MODE => pure defaults => embedded on-disk store, so a
    # native single-user install works with no separate server (and dodges the
    # 8100 collision with the SDXL diffusion server).
    monkeypatch.delenv("CHROMADB_HOST", raising=False)
    monkeypatch.delenv("CHROMADB_PORT", raising=False)
    monkeypatch.delenv("CHROMADB_MODE", raising=False)
    monkeypatch.setenv("CHROMADB_PATH", str(tmp_path / "chroma"))
    # Must NOT touch the network: any HTTP attempt would be a bug here.
    monkeypatch.setattr(cc, "_make_http_client", lambda *a, **k: pytest.fail("used HTTP"))
    client = cc.get_chroma_client()
    assert client is not None
    assert cc._client is client
    # A round-trip through the real embedded store proves it's usable.
    col = client.get_or_create_collection("mem_test")
    col.add(ids=["1"], documents=["hello"], embeddings=[[0.1, 0.2, 0.3]])
    assert col.count() == 1
    cc.reset_client()


def test_embedded_mode_forced(monkeypatch, tmp_path):
    pytest.importorskip("chromadb")
    if not _embedded_supported(tmp_path / "_probe"):
        pytest.skip("HTTP-only chromadb-client: embedded PersistentClient not usable")
    cc.reset_client()
    # CHROMADB_MODE=embedded must use the on-disk store even if server env is set.
    monkeypatch.setenv("CHROMADB_MODE", "embedded")
    monkeypatch.setenv("CHROMADB_HOST", "127.0.0.1")
    monkeypatch.setenv("CHROMADB_PORT", str(_free_port()))
    monkeypatch.setenv("CHROMADB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(cc, "_make_http_client", lambda *a, **k: pytest.fail("used HTTP"))
    assert cc.get_chroma_client() is not None
    cc.reset_client()


def test_auto_mode_falls_back_to_http_when_embedded_unavailable(monkeypatch):
    pytest.importorskip("chromadb")
    cc.reset_client()
    # Pure defaults, but the installed package can't run embedded (HTTP-only build):
    # must fall back to the HTTP client rather than crash. Models this user's env.
    monkeypatch.delenv("CHROMADB_HOST", raising=False)
    monkeypatch.delenv("CHROMADB_PORT", raising=False)
    monkeypatch.delenv("CHROMADB_MODE", raising=False)
    sentinel = object()
    monkeypatch.setattr(cc, "_make_embedded_client", lambda: None)
    monkeypatch.setattr(cc, "_make_http_client", lambda host, port: sentinel)
    assert cc.get_chroma_client() is sentinel
    cc.reset_client()
