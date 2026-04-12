"""Godot signal and event system patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class SignalPriority(int, Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

    def is_high_priority(self) -> bool:
        return self in {SignalPriority.HIGH, SignalPriority.CRITICAL}


class ConnectionFlags(str, Enum):
    DEFERRED = "deferred"
    ONE_SHOT = "one_shot"
    PERSIST = "persist"
    REFERENCE = "reference"

    def requires_cleanup(self) -> bool:
        return self != ConnectionFlags.ONE_SHOT


@dataclass
class SignalConnection:
    target: str
    method: str
    flags: set[ConnectionFlags] = field(default_factory=set)
    priority: SignalPriority = SignalPriority.NORMAL
    call_count: int = 0

    def is_one_shot(self) -> bool:
        return ConnectionFlags.ONE_SHOT in self.flags

    def is_deferred(self) -> bool:
        return ConnectionFlags.DEFERRED in self.flags

    def record_call(self) -> None:
        self.call_count += 1

    def should_disconnect(self) -> bool:
        return self.is_one_shot() and self.call_count > 0


@dataclass
class SignalDefinition:
    name: str
    parameters: list[str] = field(default_factory=list)
    description: str = ""

    def parameter_count(self) -> int:
        return len(self.parameters)

    def signature(self) -> str:
        params = ", ".join(self.parameters) if self.parameters else ""
        return f"{self.name}({params})"


class Signal:
    def __init__(self, definition: SignalDefinition) -> None:
        self._definition = definition
        self._connections: list[tuple[SignalConnection, Callable[..., Any]]] = []
        self._emit_count = 0
        self._deferred_queue: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    @property
    def name(self) -> str:
        return self._definition.name

    def connect(
        self,
        callback: Callable[..., Any],
        target: str = "",
        method: str = "",
        flags: set[ConnectionFlags] | None = None,
        priority: SignalPriority = SignalPriority.NORMAL,
    ) -> SignalConnection:
        conn = SignalConnection(
            target=target or callback.__name__,
            method=method or callback.__name__,
            flags=flags or set(),
            priority=priority,
        )
        self._connections.append((conn, callback))
        self._connections.sort(key=lambda x: x[0].priority.value, reverse=True)
        return conn

    def disconnect(self, target: str, method: str) -> bool:
        original_len = len(self._connections)
        self._connections = [
            (c, cb)
            for c, cb in self._connections
            if not (c.target == target and c.method == method)
        ]
        return len(self._connections) < original_len

    def emit(self, *args: Any, **kwargs: Any) -> int:
        self._emit_count += 1
        called = 0
        to_remove: list[int] = []
        for i, (conn, callback) in enumerate(self._connections):
            if conn.is_deferred():
                self._deferred_queue.append((args, kwargs))
            else:
                callback(*args, **kwargs)
                conn.record_call()
                called += 1
            if conn.should_disconnect():
                to_remove.append(i)
        for i in reversed(to_remove):
            self._connections.pop(i)
        return called

    def flush_deferred(self) -> int:
        flushed = 0
        while self._deferred_queue:
            args, kwargs = self._deferred_queue.pop(0)
            for conn, callback in self._connections:
                if conn.is_deferred():
                    callback(*args, **kwargs)
                    conn.record_call()
                    flushed += 1
        return flushed

    def connection_count(self) -> int:
        return len(self._connections)

    def emit_count(self) -> int:
        return self._emit_count

    def is_connected(self, target: str, method: str) -> bool:
        return any(
            c.target == target and c.method == method for c, _ in self._connections
        )


class SignalBus:
    """Global signal bus for decoupled communication."""

    def __init__(self) -> None:
        self._signals: dict[str, Signal] = {}

    def register(self, definition: SignalDefinition) -> Signal:
        sig = Signal(definition)
        self._signals[definition.name] = sig
        return sig

    def get(self, name: str) -> Signal | None:
        return self._signals.get(name)

    def emit(self, name: str, *args: Any, **kwargs: Any) -> int:
        sig = self._signals.get(name)
        if sig is None:
            return 0
        return sig.emit(*args, **kwargs)

    def signal_count(self) -> int:
        return len(self._signals)

    def total_connections(self) -> int:
        return sum(s.connection_count() for s in self._signals.values())


@dataclass
class InputEvent:
    event_type: str
    device: int = 0
    pressed: bool = False
    strength: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_action(self, action: str) -> bool:
        return self.metadata.get("action") == action

    def is_key(self, keycode: int) -> bool:
        return self.metadata.get("keycode") == keycode

    def is_mouse_button(self, button: int) -> bool:
        return (
            self.event_type == "mouse_button" and self.metadata.get("button") == button
        )


class InputMap:
    def __init__(self) -> None:
        self._actions: dict[str, list[InputEvent]] = {}
        self._just_pressed: set[str] = set()
        self._held: set[str] = set()

    def add_action(self, action: str) -> InputMap:
        if action not in self._actions:
            self._actions[action] = []
        return self

    def action_add_event(self, action: str, event: InputEvent) -> InputMap:
        if action not in self._actions:
            self._actions[action] = []
        event.metadata["action"] = action
        self._actions[action].append(event)
        return self

    def is_action_pressed(self, action: str) -> bool:
        return action in self._held

    def is_action_just_pressed(self, action: str) -> bool:
        return action in self._just_pressed

    def process_event(self, event: InputEvent) -> list[str]:
        triggered: list[str] = []
        for action, events in self._actions.items():
            if any(e.is_action(action) for e in events):
                if event.pressed:
                    self._just_pressed.add(action)
                    self._held.add(action)
                else:
                    self._held.discard(action)
                triggered.append(action)
        return triggered

    def flush_just_pressed(self) -> None:
        self._just_pressed.clear()

    def action_count(self) -> int:
        return len(self._actions)
