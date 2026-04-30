"""Demo: LLM-guided NPC decisions using local Ollama (simulated).

Shows how Python-side AI logic drives NPC personalities, FSM transitions,
and dialogue generation without touching GDScript at runtime.
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from patterns.entities import (
    ComponentType,
    Entity,
    EntityRegistry,
    EntityTag,
    HealthComponent,
    MovementComponent,
    TransformComponent,
    Vector2,
)
from patterns.scene import NodeType, Scene, SceneManager, SceneNode, SceneState
from patterns.signals import SignalBus, SignalDefinition

# ---------------------------------------------------------------------------
# NPC FSM states
# ---------------------------------------------------------------------------


class NPCState(str, Enum):
    """Finite-state machine states for an NPC."""

    IDLE = "idle"
    PATROL = "patrol"
    TALK = "talk"
    FLEE = "flee"
    ATTACK = "attack"

    def can_transition_to(self, target: NPCState) -> bool:
        """Return whether a direct FSM edge exists between the two states."""
        edges: dict[NPCState, set[NPCState]] = {
            NPCState.IDLE: {
                NPCState.PATROL,
                NPCState.TALK,
                NPCState.FLEE,
                NPCState.ATTACK,
            },
            NPCState.PATROL: {
                NPCState.IDLE,
                NPCState.TALK,
                NPCState.FLEE,
                NPCState.ATTACK,
            },
            NPCState.TALK: {NPCState.IDLE, NPCState.FLEE},
            NPCState.FLEE: {NPCState.IDLE},
            NPCState.ATTACK: {NPCState.IDLE, NPCState.FLEE},
        }
        return target in edges.get(self, set())


# ---------------------------------------------------------------------------
# NPC personality dataclass
# ---------------------------------------------------------------------------


@dataclass
class NPCPersonality:
    """Personality profile fed to the LLM prompt as system context."""

    name: str
    archetype: str  # e.g. "merchant", "guard", "sage"
    aggression: float = 0.2  # 0.0 calm … 1.0 hostile
    curiosity: float = 0.5  # willingness to initiate dialogue
    cowardice: float = 0.3  # tendency to flee when threatened
    traits: list[str] = field(default_factory=list)

    def system_prompt(self) -> str:
        """Build a concise LLM system prompt from this personality."""
        trait_text = ", ".join(self.traits) if self.traits else "none"
        return (
            f"You are {self.name}, a {self.archetype}. "
            f"Aggression={self.aggression:.1f}, curiosity={self.curiosity:.1f}, "
            f"cowardice={self.cowardice:.1f}. Traits: {trait_text}. "
            'Respond with a JSON object: {"action": <state>, "dialogue": <string>}.'
        )


# ---------------------------------------------------------------------------
# Simulated Ollama client
# ---------------------------------------------------------------------------


class SimulatedOllama:
    """Mimics a blocking Ollama /api/generate call (no network required).

    In production replace ``generate`` with an actual HTTP POST to
    ``http://localhost:11434/api/generate``.
    """

    _RESPONSE_BANK: dict[str, list[dict[str, str]]] = {
        "merchant": [
            {
                "action": "talk",
                "dialogue": "Fine wares for a fine adventurer! Step right up.",
            },
            {"action": "idle", "dialogue": "Business is slow today…"},
            {"action": "flee", "dialogue": "Bandits! Someone call the guards!"},
        ],
        "guard": [
            {"action": "patrol", "dialogue": "Nothing to see here, move along."},
            {"action": "attack", "dialogue": "Halt! You are under arrest!"},
            {"action": "idle", "dialogue": "All quiet on the eastern wall."},
        ],
        "sage": [
            {"action": "talk", "dialogue": "Seek wisdom before power, young one."},
            {"action": "idle", "dialogue": "The stars speak of coming change…"},
            {
                "action": "talk",
                "dialogue": "Have you considered the philosophy of recursion?",
            },
        ],
    }

    def generate(
        self, personality: NPCPersonality, player_action: str, latency_ms: float = 480.0
    ) -> dict[str, Any]:
        """Return a simulated LLM response with realistic latency."""
        time.sleep(latency_ms / 1_000.0)
        bank = self._RESPONSE_BANK.get(
            personality.archetype, [{"action": "idle", "dialogue": "..."}]
        )
        return random.choice(bank)


# ---------------------------------------------------------------------------
# LLM-guided NPC component
# ---------------------------------------------------------------------------


@dataclass
class LLMNPCComponent:
    """AI component that wraps an LLM call to decide NPC actions."""

    personality: NPCPersonality
    ollama: SimulatedOllama
    fsm_state: NPCState = NPCState.IDLE
    dialogue_history: list[str] = field(default_factory=list)
    response_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def decide(
        self, player_action: str, use_cache: bool = False
    ) -> tuple[NPCState, str]:
        """Query LLM and return (next_state, dialogue).  Uses cache when requested."""
        cache_key = f"{self.personality.name}:{player_action}"
        if use_cache and cache_key in self.response_cache:
            raw = self.response_cache[cache_key]
        else:
            raw = self.ollama.generate(self.personality, player_action)
            self.response_cache[cache_key] = raw

        action_str = raw.get("action", "idle")
        dialogue = raw.get("dialogue", "")

        try:
            next_state = NPCState(action_str)
        except ValueError:
            next_state = NPCState.IDLE

        if not self.fsm_state.can_transition_to(next_state):
            next_state = NPCState.IDLE

        self.fsm_state = next_state
        self.dialogue_history.append(f"[{next_state.value}] {dialogue}")
        return next_state, dialogue


# ---------------------------------------------------------------------------
# Scene assembly helpers
# ---------------------------------------------------------------------------


def build_npc_entity(
    entity_id: str,
    personality: NPCPersonality,
    position: Vector2,
    ollama: SimulatedOllama,
) -> Entity:
    """Construct an NPC Entity with all required components attached."""
    npc = Entity(entity_id=entity_id, name=personality.name)
    npc.add_tag(EntityTag.FRIENDLY)
    npc.add_component(ComponentType.TRANSFORM, TransformComponent(position=position))
    npc.add_component(ComponentType.HEALTH, HealthComponent(max_hp=80.0))
    npc.add_component(
        ComponentType.MOVEMENT, MovementComponent(speed=60.0, max_speed=120.0)
    )
    npc.add_component(
        ComponentType.AI, LLMNPCComponent(personality=personality, ollama=ollama)
    )
    npc.activate()
    return npc


def build_village_scene(registry: EntityRegistry) -> Scene:
    """Build a village Scene and attach NPC nodes to it."""
    root = SceneNode(name="Village", node_type=NodeType.NODE2D)
    npc_layer = SceneNode(name="NPCs", node_type=NodeType.NODE2D)

    for entity_id, npc_entity in [(eid, e) for eid, e in registry._entities.items()]:
        node = SceneNode(name=npc_entity.name, node_type=NodeType.CHARACTER_BODY)
        node.add_to_group("npcs")
        node.set_property("entity_id", entity_id)
        npc_layer.add_child(node)

    root.add_child(npc_layer)
    scene = Scene(name="VillageScene", root=root)
    scene.transition_to(SceneState.LOADING)
    scene.transition_to(SceneState.READY)
    return scene


# ---------------------------------------------------------------------------
# Signal wiring
# ---------------------------------------------------------------------------


def wire_signals(bus: SignalBus) -> None:
    """Register game-wide signals on the shared bus."""
    bus.register(
        SignalDefinition(
            name="npc_state_changed", parameters=["npc_id", "old_state", "new_state"]
        )
    )
    bus.register(
        SignalDefinition(name="npc_dialogue", parameters=["npc_id", "dialogue"])
    )
    bus.register(SignalDefinition(name="player_action", parameters=["action"]))


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


def run_demo() -> None:
    """Run a 3-NPC, 5-round LLM dialogue simulation."""
    print("=" * 60)
    print("  LLM NPC Demo — Godot Python Gamedev Patterns")
    print("=" * 60)

    ollama = SimulatedOllama()
    registry = EntityRegistry()
    bus = SignalBus()
    wire_signals(bus)

    # --- Define three NPCs with distinct personalities ---
    personalities = [
        NPCPersonality(
            name="Aldric",
            archetype="merchant",
            aggression=0.1,
            curiosity=0.8,
            cowardice=0.6,
            traits=["greedy", "talkative", "friendly"],
        ),
        NPCPersonality(
            name="Serath",
            archetype="guard",
            aggression=0.7,
            curiosity=0.3,
            cowardice=0.1,
            traits=["disciplined", "suspicious", "loyal"],
        ),
        NPCPersonality(
            name="Orin",
            archetype="sage",
            aggression=0.05,
            curiosity=0.95,
            cowardice=0.4,
            traits=["wise", "cryptic", "patient"],
        ),
    ]

    positions = [Vector2(100.0, 200.0), Vector2(300.0, 150.0), Vector2(500.0, 220.0)]
    npcs: list[tuple[Entity, LLMNPCComponent]] = []

    for idx, (personality, pos) in enumerate(zip(personalities, positions)):
        entity = build_npc_entity(f"npc_{idx}", personality, pos, ollama)
        registry.register(entity)
        ai_comp: LLMNPCComponent = entity.get_component(ComponentType.AI)
        npcs.append((entity, ai_comp))
        print(
            f"\n[SPAWN] {personality.name} ({personality.archetype}) at {pos.x},{pos.y}"
        )

    scene = build_village_scene(registry)
    manager = SceneManager()
    manager.register(scene)
    print(
        f"\n[SCENE] '{scene.name}' ready — {scene.total_node_count()} nodes, "
        f"{registry.alive_count()} living NPCs"
    )

    # --- Simulate 5 player interactions ---
    player_actions = ["approach", "speak", "threaten", "trade", "leave"]

    print("\n" + "-" * 60)
    for round_idx, action in enumerate(player_actions, start=1):
        print(f"\n[ROUND {round_idx}] Player action: '{action}'")
        bus.emit("player_action", action)

        for entity, ai_comp in npcs:
            old_state = ai_comp.fsm_state
            # Use cache on even rounds to demonstrate the ~10 ms path
            use_cache = round_idx % 2 == 0
            new_state, dialogue = ai_comp.decide(action, use_cache=use_cache)

            bus.emit(
                "npc_state_changed", entity.entity_id, old_state.value, new_state.value
            )
            bus.emit("npc_dialogue", entity.entity_id, dialogue)

            cached_label = " (cached)" if use_cache else ""
            print(
                f"  {entity.name:10s} {old_state.value:8s} -> {new_state.value:8s}{cached_label}"
            )
            print(f'             "{dialogue}"')

    print("\n" + "=" * 60)
    print(f"[DONE] Scene node count : {scene.total_node_count()}")
    print(f"[DONE] Registry alive   : {registry.alive_count()}")
    print(
        f"[DONE] Signal bus size  : {bus.signal_count()} signals / {bus.total_connections()} connections"
    )
    npc_nodes = scene.get_nodes_in_group("npcs")
    print(f"[DONE] NPC nodes        : {len(npc_nodes)}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
