"""Tests for signal and event system patterns."""

from __future__ import annotations

import pytest

from patterns.signals import (
    ConnectionFlags,
    InputEvent,
    InputMap,
    Signal,
    SignalBus,
    SignalConnection,
    SignalDefinition,
    SignalPriority,
)


class TestSignalPriority:
    def test_is_high_priority(self):
        assert SignalPriority.HIGH.is_high_priority()
        assert SignalPriority.CRITICAL.is_high_priority()
        assert not SignalPriority.NORMAL.is_high_priority()
        assert not SignalPriority.LOW.is_high_priority()


class TestConnectionFlags:
    def test_requires_cleanup(self):
        assert ConnectionFlags.DEFERRED.requires_cleanup()
        assert ConnectionFlags.PERSIST.requires_cleanup()
        assert not ConnectionFlags.ONE_SHOT.requires_cleanup()


class TestSignalConnection:
    def test_one_shot(self):
        conn = SignalConnection(
            target="obj", method="on_event", flags={ConnectionFlags.ONE_SHOT}
        )
        assert conn.is_one_shot()
        assert not conn.should_disconnect()
        conn.record_call()
        assert conn.should_disconnect()

    def test_deferred(self):
        conn = SignalConnection(
            target="obj", method="on_event", flags={ConnectionFlags.DEFERRED}
        )
        assert conn.is_deferred()

    def test_call_count(self):
        conn = SignalConnection(target="obj", method="on_event")
        assert conn.call_count == 0
        conn.record_call()
        conn.record_call()
        assert conn.call_count == 2


class TestSignalDefinition:
    def test_signature_no_params(self):
        defn = SignalDefinition(name="ready")
        assert defn.signature() == "ready()"
        assert defn.parameter_count() == 0

    def test_signature_with_params(self):
        defn = SignalDefinition(name="hit", parameters=["damage", "source"])
        assert defn.signature() == "hit(damage, source)"
        assert defn.parameter_count() == 2


class TestSignal:
    def test_connect_and_emit(self):
        defn = SignalDefinition(name="pressed")
        sig = Signal(defn)
        calls = []
        sig.connect(lambda: calls.append(1), target="btn", method="on_pressed")
        count = sig.emit()
        assert count == 1
        assert len(calls) == 1

    def test_emit_with_args(self):
        defn = SignalDefinition(name="health_changed", parameters=["value"])
        sig = Signal(defn)
        received = []
        sig.connect(lambda v: received.append(v), target="ui", method="on_hp")
        sig.emit(42)
        assert received == [42]

    def test_disconnect(self):
        defn = SignalDefinition(name="event")
        sig = Signal(defn)
        calls = []
        sig.connect(lambda: calls.append(1), target="obj", method="on_event")
        sig.emit()
        assert len(calls) == 1
        sig.disconnect("obj", "on_event")
        sig.emit()
        assert len(calls) == 1

    def test_one_shot_auto_disconnect(self):
        defn = SignalDefinition(name="once")
        sig = Signal(defn)
        calls = []
        sig.connect(
            lambda: calls.append(1),
            target="obj",
            method="on_once",
            flags={ConnectionFlags.ONE_SHOT},
        )
        sig.emit()
        sig.emit()
        assert len(calls) == 1
        assert sig.connection_count() == 0

    def test_priority_ordering(self):
        defn = SignalDefinition(name="ordered")
        sig = Signal(defn)
        order = []
        sig.connect(
            lambda: order.append("normal"),
            target="a",
            method="m",
            priority=SignalPriority.NORMAL,
        )
        sig.connect(
            lambda: order.append("high"),
            target="b",
            method="m",
            priority=SignalPriority.HIGH,
        )
        sig.connect(
            lambda: order.append("low"),
            target="c",
            method="m",
            priority=SignalPriority.LOW,
        )
        sig.emit()
        assert order[0] == "high"
        assert order[-1] == "low"

    def test_deferred_queue(self):
        defn = SignalDefinition(name="deferred_event")
        sig = Signal(defn)
        calls = []
        sig.connect(
            lambda: calls.append(1),
            target="obj",
            method="on_event",
            flags={ConnectionFlags.DEFERRED},
        )
        count = sig.emit()
        assert count == 0
        assert len(calls) == 0
        flushed = sig.flush_deferred()
        assert flushed == 1

    def test_is_connected(self):
        defn = SignalDefinition(name="test")
        sig = Signal(defn)
        sig.connect(lambda: None, target="obj", method="cb")
        assert sig.is_connected("obj", "cb")
        assert not sig.is_connected("obj", "other")

    def test_emit_count(self):
        defn = SignalDefinition(name="counter")
        sig = Signal(defn)
        sig.emit()
        sig.emit()
        assert sig.emit_count() == 2


class TestSignalBus:
    def test_register_and_emit(self):
        bus = SignalBus()
        defn = SignalDefinition(name="game_over")
        bus.register(defn)
        calls = []
        sig = bus.get("game_over")
        assert sig is not None
        sig.connect(lambda: calls.append(1), target="ui", method="show_game_over")
        bus.emit("game_over")
        assert len(calls) == 1

    def test_emit_missing(self):
        bus = SignalBus()
        result = bus.emit("nonexistent")
        assert result == 0

    def test_total_connections(self):
        bus = SignalBus()
        s1 = bus.register(SignalDefinition(name="s1"))
        s2 = bus.register(SignalDefinition(name="s2"))
        s1.connect(lambda: None, target="a", method="m")
        s1.connect(lambda: None, target="b", method="m")
        s2.connect(lambda: None, target="c", method="m")
        assert bus.total_connections() == 3
        assert bus.signal_count() == 2


class TestInputEvent:
    def test_is_action(self):
        event = InputEvent(event_type="key", pressed=True)
        event.metadata["action"] = "jump"
        assert event.is_action("jump")
        assert not event.is_action("attack")

    def test_is_key(self):
        event = InputEvent(event_type="key", pressed=True)
        event.metadata["keycode"] = 32
        assert event.is_key(32)
        assert not event.is_key(65)

    def test_is_mouse_button(self):
        event = InputEvent(event_type="mouse_button", pressed=True)
        event.metadata["button"] = 1
        assert event.is_mouse_button(1)
        assert not event.is_mouse_button(2)


class TestInputMap:
    def test_add_action(self):
        imap = InputMap()
        imap.add_action("jump")
        assert imap.action_count() == 1

    def test_process_press(self):
        imap = InputMap()
        imap.add_action("jump")
        event = InputEvent(event_type="key", pressed=True)
        event.metadata["action"] = "jump"
        imap.action_add_event("jump", event)
        imap.process_event(event)
        assert imap.is_action_pressed("jump")
        assert imap.is_action_just_pressed("jump")

    def test_flush_just_pressed(self):
        imap = InputMap()
        imap.add_action("jump")
        event = InputEvent(event_type="key", pressed=True)
        event.metadata["action"] = "jump"
        imap.action_add_event("jump", event)
        imap.process_event(event)
        imap.flush_just_pressed()
        assert not imap.is_action_just_pressed("jump")
        assert imap.is_action_pressed("jump")

    def test_process_release(self):
        imap = InputMap()
        imap.add_action("jump")
        press = InputEvent(event_type="key", pressed=True)
        press.metadata["action"] = "jump"
        imap.action_add_event("jump", press)
        imap.process_event(press)
        release = InputEvent(event_type="key", pressed=False)
        release.metadata["action"] = "jump"
        imap.process_event(release)
        assert not imap.is_action_pressed("jump")


@pytest.mark.asyncio
async def test_signal_bus_async():
    bus = SignalBus()
    bus.register(SignalDefinition(name="async_event"))
    sig = bus.get("async_event")
    assert sig is not None
    results = []
    sig.connect(lambda v: results.append(v), target="x", method="cb")
    bus.emit("async_event", 99)
    assert results == [99]
