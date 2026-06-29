# core/models.py
"""
Pure data models — no database logic, no side effects.

These are simple datacontainers. All persistence is handled by SessionManager.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .session_manager import SessionManager

# Module-level session manager singleton (single source of truth)
_SESSION_MANAGER_INSTANCE: Optional["SessionManager"] = None


def set_session_manager_instance(manager: "SessionManager"):
    """Set the global SessionManager singleton."""
    global _SESSION_MANAGER_INSTANCE
    _SESSION_MANAGER_INSTANCE = manager


def get_session_manager_instance() -> Optional["SessionManager"]:
    """Get the global SessionManager singleton."""
    return _SESSION_MANAGER_INSTANCE


# Keep legacy name for backward compatibility
set_session_manager = set_session_manager_instance
get_session_manager = get_session_manager_instance


def _context_note_from_tool_events(tool_events: Any) -> str:
    """Compact, model-facing record of what tools the assistant ran on a turn.

    On history reload the LLM only sees each turn's ``role``/``content`` — a tool's
    results live in ``metadata.tool_events``, which the model never sees. So after
    the agent generates an image and replies with prose offering tweaks, a
    follow-up like "make it colored" arrives with NO record of what was generated
    (not even the prompt), and the model treats it as a fresh, contextless request.

    This surfaces a short note the model CAN read. Image tools carry the prompt
    (the essential memory); other tools get just name + command (no output echo)
    so coding/file sessions aren't bloated. The whole note is length-capped.
    """
    if not tool_events or not isinstance(tool_events, (list, tuple)):
        return ""
    lines: List[str] = []
    for ev in tool_events:
        if not isinstance(ev, dict):
            continue
        tool = str(ev.get("tool") or "tool").strip()
        if ev.get("image_prompt") or ev.get("image_url") or tool in ("generate_image", "edit_image"):
            prompt = str(ev.get("image_prompt") or ev.get("command") or "").strip()
            extras = [str(ev[k]) for k in ("image_model", "image_size") if ev.get(k)]
            meta = f" ({', '.join(extras)})" if extras else ""
            if prompt:
                lines.append(f"- {tool}: generated an image{meta} — prompt: {prompt[:400]}")
            else:
                lines.append(f"- {tool}: generated an image{meta}")
        else:
            cmd = str(ev.get("command") or "").strip()
            lines.append(f"- {tool}: {cmd[:160]}" if cmd else f"- {tool}")
    if not lines:
        return ""
    note = "[Earlier this turn I used these tools:\n" + "\n".join(lines) + "]"
    return note[:1200]


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API responses."""
        result = {"role": self.role, "content": self.content}
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def get(self, key: str, default=None):
        """Dict-like access for compatibility."""
        return getattr(self, key, default)


@dataclass
class Session:
    """A chat session — pure data container.

    ``.history`` is the authoritative mutable message list. Callers may
    read, append, pop, or reassign it directly — these changes take
    effect immediately. ``_history`` remains a compatibility alias that
    always resolves to the authoritative ``history`` list.

    Each session gets its own unique history list at construction time
    (the dataclass default is never shared between instances).
    """

    id: str
    name: str
    endpoint_url: str
    model: str
    rag: bool = False
    archived: bool = False
    headers: Optional[Dict[str, str]] = None
    history: List[ChatMessage] = None
    owner: Optional[str] = None
    is_important: bool = False
    message_count: int = 0

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        # Ensure each session gets its OWN list (not the shared dataclass default)
        if self.history is None:
            self.history = []

    @property
    def _history(self) -> List[ChatMessage]:
        """Compatibility alias for callers that still reference ``_history``."""
        return self.history

    @_history.setter
    def _history(self, messages: List[ChatMessage]):
        self.history = messages

    def add_message(self, message: ChatMessage):
        """
        Add a message to this session.

        Appends to the authoritative history list and increments
        message_count. Delegates to SessionManager for persistence
        if available.
        """
        self.history.append(message)
        self.message_count = len(self.history)

        # Delegate to session manager for persistence
        if _SESSION_MANAGER_INSTANCE:
            _SESSION_MANAGER_INSTANCE._persist_message(self.id, message)

    def get_context_messages(self) -> List[Dict[str, Any]]:
        """Get messages in format for LLM API.

        Slash-command / setup replies are persisted to history so they render
        in the transcript, but they are UI chatter (e.g. ``/setup ...`` and its
        status lines) the user never meant as conversation. They carry
        ``metadata.source == "slash"``; exclude them here so they never reach
        the model. Display/history-load paths use the raw ``history`` and are
        unaffected.
        """
        out: List[Dict[str, Any]] = []
        for msg in self.history:
            md = msg.metadata or {}
            if md.get("source") == "slash":
                continue
            d = msg.to_dict()
            # Replay a compact record of the assistant's tool actions (esp. an
            # image's prompt) into the content the model can actually read, so
            # multi-turn edits ("make it colored") keep context. See
            # _context_note_from_tool_events.
            if msg.role == "assistant":
                note = _context_note_from_tool_events(md.get("tool_events"))
                if note:
                    content = d.get("content")
                    if isinstance(content, str) and content.strip():
                        d["content"] = content + "\n\n" + note
                    elif isinstance(content, list):
                        d["content"] = content + [{"type": "text", "text": note}]
                    else:
                        d["content"] = note
            out.append(d)
        return out

    def get(self, key: str, default=None):
        """Dict-like access for compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        """Allow session['field'] syntax."""
        return getattr(self, key)
