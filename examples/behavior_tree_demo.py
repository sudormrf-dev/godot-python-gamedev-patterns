"""Demo: Behavior tree for a guardian NPC.

Implements Selector / Sequence / Leaf nodes using stdlib only.
Simulates 10 gameplay turns and prints the BT tick result each frame.
"""

from __future__ import annotations

import os
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from patterns.entities import (
    ComponentType,
    Entity,
    EntityTag,
    HealthComponent,
    MovementComponent,
    TransformComponent,
    Vector2,
)
from patterns.signals import SignalBus, SignalDefinition

# ---------------------------------------------------------------------------
# Behavior tree primitives
# ---------------------------------------------------------------------------


class BTStatus(str, Enum):
    """Return value of every behavior tree node tick."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class BTNode:
    """Abstract base for all behavior tree nodes."""

    def tick(self, ctx: BTContext) -> BTStatus:
        """Execute one tick; return SUCCESS, FAILURE, or RUNNING."""
        raise NotImplementedError


class BTLeaf(BTNode):
    """Leaf node backed by a plain Python callable."""

    def __init__(self, name: str, fn: Callable[[BTContext], BTStatus]) -> None:
        self.name = name
        self._fn = fn

    def tick(self, ctx: BTContext) -> BTStatus:
        status = self._fn(ctx)
        return status


class BTSequence(BTNode):
    """Runs children left-to-right; fails on first FAILURE (AND semantics)."""

    def __init__(self, children: list[BTNode]) -> None:
        self.children = children

    def tick(self, ctx: BTContext) -> BTStatus:
        for child in self.children:
            status = child.tick(ctx)
            if status != BTStatus.SUCCESS:
                return status
        return BTStatus.SUCCESS


class BTSelector(BTNode):
    """Runs children left-to-right; succeeds on first SUCCESS (OR semantics)."""

    def __init__(self, children: list[BTNode]) -> None:
        self.children = children

    def tick(self, ctx: BTContext) -> BTStatus:
        for child in self.children:
            status = child.tick(ctx)
            if status != BTStatus.FAILURE:
                return status
        return BTStatus.FAILURE


# ---------------------------------------------------------------------------
# Shared context passed to every tick
# ---------------------------------------------------------------------------


@dataclass
class BTContext:
    """Mutable context shared across all BT nodes for a single tick."""

    guardian: Entity
    player_position: Vector2
    player_visible: bool = False
    in_attack_range: bool = False
    patrol_index: int = 0
    patrol_waypoints: list[Vector2] = field(default_factory=list)
    log: list[str] = field(default_factory=list)

    def guardian_position(self) -> Vector2:
        tf: TransformComponent | None = self.guardian.get_component(
            ComponentType.TRANSFORM
        )
        return tf.position if tf else Vector2()

    def distance_to_player(self) -> float:
        return self.guardian_position().distance_to(self.player_position)

    def record(self, msg: str) -> None:
        self.log.append(msg)


# ---------------------------------------------------------------------------
# Leaf implementations
# ---------------------------------------------------------------------------


def check_line_of_sight(ctx: BTContext) -> BTStatus:
    """Return SUCCESS if the player is within LOS distance (< 200 units)."""
    dist = ctx.distance_to_player()
    visible = dist < 200.0
    ctx.player_visible = visible
    ctx.record(f"  LOS check: dist={dist:.1f} visible={visible}")
    return BTStatus.SUCCESS if visible else BTStatus.FAILURE


def move_towards_player(ctx: BTContext) -> BTStatus:
    """Step guardian toward player; return RUNNING until in attack range."""
    dist = ctx.distance_to_player()
    mv: MovementComponent | None = ctx.guardian.get_component(ComponentType.MOVEMENT)
    speed = mv.speed if mv else 80.0
    tf: TransformComponent | None = ctx.guardian.get_component(ComponentType.TRANSFORM)

    if dist < 50.0:
        ctx.in_attack_range = True
        ctx.record(f"  Move: already in attack range (dist={dist:.1f})")
        return BTStatus.SUCCESS

    if tf:
        px, py = ctx.player_position.x, ctx.player_position.y
        gx, gy = tf.position.x, tf.position.y
        step = min(speed, dist)
        ratio = step / dist
        tf.position = Vector2(gx + (px - gx) * ratio, gy + (py - gy) * ratio)

    ctx.record(f"  Move: advancing toward player (dist={dist:.1f})")
    return BTStatus.RUNNING


def perform_attack(ctx: BTContext) -> BTStatus:
    """Deal damage to the player (simulated via signal emission)."""
    if not ctx.in_attack_range:
        ctx.record("  Attack: out of range — FAILURE")
        return BTStatus.FAILURE

    hp: HealthComponent | None = ctx.guardian.get_component(ComponentType.HEALTH)
    dmg = 10.0
    ctx.record(
        f"  Attack: STRIKE for {dmg} dmg! Guardian HP={hp.current_hp if hp else '?'}"
    )
    return BTStatus.SUCCESS


def patrol(ctx: BTContext) -> BTStatus:
    """Advance to next waypoint in the patrol route."""
    if not ctx.patrol_waypoints:
        ctx.record("  Patrol: no waypoints — FAILURE")
        return BTStatus.FAILURE

    wp = ctx.patrol_waypoints[ctx.patrol_index % len(ctx.patrol_waypoints)]
    tf: TransformComponent | None = ctx.guardian.get_component(ComponentType.TRANSFORM)
    if tf:
        tf.position = wp

    ctx.patrol_index += 1
    ctx.record(
        f"  Patrol: moved to waypoint {ctx.patrol_index} ({wp.x:.0f},{wp.y:.0f})"
    )
    return BTStatus.SUCCESS


def chase_player(ctx: BTContext) -> BTStatus:
    """Pursue the player when visible but not yet in attack range."""
    if not ctx.player_visible:
        ctx.record("  Chase: player not visible — FAILURE")
        return BTStatus.FAILURE

    dist = ctx.distance_to_player()
    if dist < 50.0:
        ctx.in_attack_range = True
        ctx.record(f"  Chase: caught up with player (dist={dist:.1f})")
        return BTStatus.SUCCESS

    ctx.record(f"  Chase: pursuing player (dist={dist:.1f})")
    return move_towards_player(ctx)


# ---------------------------------------------------------------------------
# BT factory
# ---------------------------------------------------------------------------


def build_guardian_bt() -> BTSelector:
    """Construct the guardian behavior tree.

    Tree structure:
        Selector
        ├── Sequence: attack branch
        │   ├── check_line_of_sight
        │   ├── move_towards_player
        │   └── perform_attack
        ├── chase_player
        └── patrol
    """
    attack_sequence = BTSequence(
        children=[
            BTLeaf("check_los", check_line_of_sight),
            BTLeaf("move_towards", move_towards_player),
            BTLeaf("attack", perform_attack),
        ]
    )
    return BTSelector(
        children=[
            attack_sequence,
            BTLeaf("chase_player", chase_player),
            BTLeaf("patrol", patrol),
        ]
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def build_guardian() -> Entity:
    """Construct the guardian Entity with all required components."""
    guardian = Entity(entity_id="guardian_01", name="Guardian")
    guardian.add_tag(EntityTag.ENEMY)
    guardian.add_component(
        ComponentType.TRANSFORM, TransformComponent(position=Vector2(300.0, 300.0))
    )
    guardian.add_component(
        ComponentType.HEALTH, HealthComponent(max_hp=200.0, armor=5.0)
    )
    guardian.add_component(
        ComponentType.MOVEMENT, MovementComponent(speed=80.0, max_speed=160.0)
    )
    guardian.activate()
    return guardian


def build_waypoints() -> list[Vector2]:
    """Return a small patrol circuit."""
    return [
        Vector2(100.0, 100.0),
        Vector2(500.0, 100.0),
        Vector2(500.0, 500.0),
        Vector2(100.0, 500.0),
    ]


def run_demo() -> None:
    """Simulate 10 turns of guardian behavior and print BT tick results."""
    print("=" * 60)
    print("  Behavior Tree Demo — Guardian NPC")
    print("=" * 60)

    guardian = build_guardian()
    bt = build_guardian_bt()

    bus = SignalBus()
    bus.register(SignalDefinition("bt_tick", parameters=["turn", "status"]))

    waypoints = build_waypoints()
    ctx = BTContext(
        guardian=guardian,
        player_position=Vector2(999.0, 999.0),  # player starts far away
        patrol_waypoints=waypoints,
    )

    random.seed(42)
    for turn in range(1, 11):
        ctx.log.clear()
        ctx.in_attack_range = False
        ctx.player_visible = False

        # Randomly move player closer every few turns to trigger chase/attack
        if turn >= 5:
            ctx.player_position = Vector2(
                300.0 + random.uniform(-180.0, 180.0),
                300.0 + random.uniform(-180.0, 180.0),
            )

        status = bt.tick(ctx)
        bus.emit("bt_tick", turn, status.value)

        print(
            f"\n[TURN {turn:02d}] player@({ctx.player_position.x:.0f},{ctx.player_position.y:.0f})"
            f"  ->  BT result: {status.value}"
        )
        for line in ctx.log:
            print(line)

    hp: HealthComponent = guardian.get_component(ComponentType.HEALTH)
    print(f"\n[END] Guardian HP: {hp.current_hp:.1f}/{hp.max_hp:.1f}")
    print(
        f"[END] Guardian position: ({ctx.guardian_position().x:.1f}, {ctx.guardian_position().y:.1f})"
    )
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
