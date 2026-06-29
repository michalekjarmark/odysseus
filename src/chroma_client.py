"""
chroma_client.py

Singleton ChromaDB client. Talks to a standalone ChromaDB service over HTTP when
one is configured (e.g. Docker compose), or falls back to an embedded on-disk
PersistentClient for a native single-user install that runs no separate server.
See ``get_chroma_client`` for the mode-selection rules (CHROMADB_MODE).
"""

import os
import socket
import logging

logger = logging.getLogger(__name__)

_client = None

# A short connect probe so an unreachable ChromaDB fails fast instead of
# blocking on the OS connection timeout (~30-60s, WinError 10060 on Windows),
# which otherwise stalls app startup. Tunable via CHROMADB_CONNECT_TIMEOUT.
_CONNECT_TIMEOUT = float(os.getenv("CHROMADB_CONNECT_TIMEOUT", "2.0"))


def _port_open(host: str, port: int, timeout: float = None) -> bool:
    """Return True if a TCP connection to host:port succeeds within timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout or _CONNECT_TIMEOUT):
            return True
    except OSError:
        return False


def _embedded_path() -> str:
    """On-disk location for the embedded (PersistentClient) ChromaDB store."""
    explicit = os.getenv("CHROMADB_PATH")
    if explicit:
        return explicit
    try:
        from src.constants import DATA_DIR  # lazy: keep this module import-light
        base = str(DATA_DIR)
    except Exception:
        base = "data"
    return os.path.join(base, "chroma")


def _make_embedded_client():
    """Create an embedded on-disk ChromaDB client, or None if unavailable.

    Uses ``chromadb.PersistentClient`` so a single-user / native install needs no
    separate ChromaDB server. Returns None when the embedded backend isn't usable —
    either ``PersistentClient`` is missing, or the installed package is the
    lightweight HTTP-only ``chromadb-client`` (which exposes the symbol but raises
    "Chroma is running in http-only client mode" on use). In both cases the caller
    falls back to the remote path instead of crashing.
    """
    import chromadb
    make = getattr(chromadb, "PersistentClient", None)
    if make is None:
        return None
    path = _embedded_path()
    try:
        os.makedirs(path, exist_ok=True)
        client = make(path=path)
        client.heartbeat()  # cheap validity check before caching
    except Exception as e:
        # HTTP-only `chromadb-client` build, or a broken store: don't crash —
        # let the caller fall back to HTTP. Surface the actionable hint once.
        logger.warning(
            "Embedded ChromaDB unavailable (%s). For a server-less native install, "
            "install the full package: pip uninstall chromadb-client -y && "
            "pip install chromadb", e.__class__.__name__,
        )
        return None
    logger.info(f"ChromaDB embedded (PersistentClient) at {path}")
    return client


def _make_http_client(host: str, port: int):
    """Create a remote HTTP ChromaDB client, raising a clear error if unreachable."""
    import chromadb
    if not _port_open(host, port):
        raise RuntimeError(
            f"ChromaDB is not reachable at {host}:{port}. Start the ChromaDB "
            f"service (e.g. `docker compose up chromadb`) or set CHROMADB_HOST / "
            f"CHROMADB_PORT to point at a running instance."
        )
    client = chromadb.HttpClient(host=host, port=port)
    # Health check before caching — if the port is open but the service isn't
    # healthy yet (e.g. still starting), don't poison the singleton with a dead
    # client; leave _client unset so the next call retries.
    client.heartbeat()
    logger.info(f"ChromaDB connected: {host}:{port}")
    return client


def get_chroma_client():
    """Get or create the singleton ChromaDB client.

    Mode is chosen by ``CHROMADB_MODE`` (``auto`` default / ``http`` / ``embedded``):

    - ``http``: always talk to a remote ChromaDB server (legacy behaviour).
    - ``embedded``: always use an on-disk ``PersistentClient`` (no server needed).
    - ``auto``: if the operator pointed us at a server (``CHROMADB_HOST`` or
      ``CHROMADB_PORT`` set — e.g. Docker compose sets these), use HTTP and surface
      a down server as an error. Otherwise (pure defaults — a typical native
      single-user install with no separate server, where the default port 8100 also
      collides with the SDXL diffusion server) use an embedded on-disk store so
      user-memory vectors and RAG work with zero extra setup. If PersistentClient
      isn't available (HTTP-only ``chromadb-client`` package), fall back to the HTTP
      path and its clear "start the service" error.

    Raises RuntimeError with a clear install hint if the `chromadb` package
    is not installed — it's an optional dependency (RAG + memory vectors).
    """
    global _client
    if _client is not None:
        return _client

    try:
        import chromadb  # noqa: F401  (presence check; helpers import it lazily)
    except ImportError as e:
        raise RuntimeError(
            "ChromaDB integration is not installed. Install the optional "
            "dependency with: pip install chromadb-client"
        ) from e

    host = os.getenv("CHROMADB_HOST", "localhost")
    port = int(os.getenv("CHROMADB_PORT", "8100"))
    mode = (os.getenv("CHROMADB_MODE", "auto") or "auto").strip().lower()
    explicit_server = bool(os.getenv("CHROMADB_HOST") or os.getenv("CHROMADB_PORT"))

    if mode == "embedded":
        client = _make_embedded_client()
        if client is None:
            raise RuntimeError(
                "CHROMADB_MODE=embedded but PersistentClient is unavailable — "
                "install the full `chromadb` package (not just `chromadb-client`)."
            )
        _client = client
        return _client

    if mode == "http" or explicit_server:
        # Operator explicitly pointed us at a server: honour it strictly so a
        # down/misconfigured server is a real error, never silently masked by a
        # divergent local store.
        _client = _make_http_client(host, port)
        return _client

    # auto + pure defaults: prefer an embedded store; fall back to HTTP only if
    # PersistentClient isn't available.
    client = _make_embedded_client()
    if client is None:
        client = _make_http_client(host, port)
    _client = client
    return _client


def reset_client():
    """Reset the singleton (e.g. after config change)."""
    global _client
    _client = None
