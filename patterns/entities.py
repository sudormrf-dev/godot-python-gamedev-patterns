"""Godot entity-component patterns for game objects."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ComponentType(str, Enum):
    TRANSFORM = "transform"
    HEALTH = "health"
    MOVEMENT = "movement"
    COLLISION = "collision"
    SPRITE = "sprite"
    ANIMATION = "animation"
    AI = "ai"
    INVENTORY = "inventory"
    STATS = "stats"
    AUDIO = "audio"

    def is_physics(self) -> bool:
        return self in {ComponentType.MOVEMENT, ComponentType.COLLISION}

    def is_visual(self) -> bool:
        return self in {ComponentType.SPRITE, ComponentType.ANIMATION}


class EntityState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    DYING = "dying"
    DEAD = "dead"
    SPAWNING = "spawning"

    def is_alive(self) -> bool:
        return self in {EntityState.IDLE, EntityState.ACTIVE, EntityState.SPAWNING}

    def can_interact(self) -> bool:
        return self == EntityState.ACTIVE


class EntityTag(str, Enum):
    PLAYER = "player"
    ENEMY = "enemy"
    FRIENDLY = "friendly"
    PROJECTILE = "projectile"
    PICKUP = "pickup"
    OBSTACLE = "obstacle"
    TRIGGER = "trigger"
    BOSS = "boss"

    def is_combatant(self) -> bool:
        return self in {EntityTag.PLAYER, EntityTag.ENEMY, EntityTag.BOSS}

    def is_collectible(self) -> bool:
        return self == EntityTag.PICKUP


@dataclass
class Vector2:
    x: float = 0.0
    y: float = 0.0

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y

    def length(self) -> float:
        return math.sqrt(self.length_squared())

    def distance_to(self, other: Vector2) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)

    def add(self, other: Vector2) -> Vector2:
        return Vector2(self.x + other.x, self.y + other.y)

    def scale(self, factor: float) -> Vector2:
        return Vector2(self.x * factor, self.y * factor)

    def is_zero(self) -> bool:
        return self.x == 0.0 and self.y == 0.0


@dataclass
class TransformComponent:
    position: Vector2 = field(default_factory=Vector2)
    rotation: float = 0.0
    scale: Vector2 = field(default_factory=lambda: Vector2(1.0, 1.0))

    def translate(self, delta: Vector2) -> TransformComponent:
        self.position = self.position.add(delta)
        return self

    def rotate(self, angle: float) -> TransformComponent:
        self.rotation += angle
        return self

    def is_flipped_h(self) -> bool:
        return self.scale.x < 0.0

    def is_flipped_v(self) -> bool:
        return self.scale.y < 0.0


@dataclass
class HealthComponent:
    max_hp: float
    current_hp: float = 0.0
    armor: float = 0.0
    regen_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.current_hp == 0.0:
            self.current_hp = self.max_hp

    def take_damage(self, amount: float) -> float:
        actual = max(0.0, amount - self.armor)
        self.current_hp = max(0.0, self.current_hp - actual)
        return actual

    def heal(self, amount: float) -> float:
        before = self.current_hp
        self.current_hp = min(self.max_hp, self.current_hp + amount)
        return self.current_hp - before

    def is_alive(self) -> bool:
        return self.current_hp > 0.0

    def is_full_hp(self) -> bool:
        return self.current_hp >= self.max_hp

    def hp_percent(self) -> float:
        if self.max_hp <= 0.0:
            return 0.0
        return self.current_hp / self.max_hp

    def tick_regen(self, delta: float) -> float:
        if self.regen_rate > 0.0 and not self.is_full_hp():
            return self.heal(self.regen_rate * delta)
        return 0.0


@dataclass
class MovementComponent:
    speed: float = 100.0
    max_speed: float = 200.0
    acceleration: float = 400.0
    friction: float = 0.8
    velocity: Vector2 = field(default_factory=Vector2)
    on_ground: bool = False

    def apply_force(self, force: Vector2) -> MovementComponent:
        self.velocity = self.velocity.add(force)
        if self.velocity.length() > self.max_speed:
            factor = self.max_speed / self.velocity.length()
            self.velocity = self.velocity.scale(factor)
        return self

    def apply_friction(self) -> MovementComponent:
        self.velocity = self.velocity.scale(self.friction)
        return self

    def is_moving(self) -> bool:
        return not self.velocity.is_zero()

    def speed_ratio(self) -> float:
        if self.max_speed <= 0.0:
            return 0.0
        return min(1.0, self.velocity.length() / self.max_speed)


class Entity:
    def __init__(self, entity_id: str, name: str) -> None:
        self._id = entity_id
        self._name = name
        self._state = EntityState.SPAWNING
        self._tags: set[EntityTag] = set()
        self._components: dict[ComponentType, Any] = {}

    @property
    def entity_id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> EntityState:
        return self._state

    def add_tag(self, tag: EntityTag) -> Entity:
        self._tags.add(tag)
        return self

    def has_tag(self, tag: EntityTag) -> bool:
        return tag in self._tags

    def tags(self) -> frozenset[EntityTag]:
        return frozenset(self._tags)

    def add_component(self, component_type: ComponentType, component: Any) -> Entity:
        self._components[component_type] = component
        return self

    def get_component(self, component_type: ComponentType) -> Any:
        return self._components.get(component_type)

    def has_component(self, component_type: ComponentType) -> bool:
        return component_type in self._components

    def activate(self) -> bool:
        if self._state == EntityState.SPAWNING:
            self._state = EntityState.ACTIVE
            return True
        return False

    def kill(self) -> bool:
        if self._state.is_alive():
            self._state = EntityState.DYING
            return True
        return False

    def destroy(self) -> None:
        self._state = EntityState.DEAD

    def is_alive(self) -> bool:
        return self._state.is_alive()

    def component_count(self) -> int:
        return len(self._components)


class EntityRegistry:
    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._by_tag: dict[EntityTag, list[str]] = {}

    def register(self, entity: Entity) -> EntityRegistry:
        self._entities[entity.entity_id] = entity
        for tag in entity.tags():
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            self._by_tag[tag].append(entity.entity_id)
        return self

    def get(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def get_by_tag(self, tag: EntityTag) -> list[Entity]:
        ids = self._by_tag.get(tag, [])
        return [e for eid in ids if (e := self._entities.get(eid)) is not None]

    def remove(self, entity_id: str) -> bool:
        entity = self._entities.pop(entity_id, None)
        if entity is None:
            return False
        for ids in self._by_tag.values():
            if entity_id in ids:
                ids.remove(entity_id)
        return True

    def alive_count(self) -> int:
        return sum(1 for e in self._entities.values() if e.is_alive())

    def total_count(self) -> int:
        return len(self._entities)

    def cleanup_dead(self) -> int:
        dead_ids = [
            eid for eid, e in self._entities.items() if e.state == EntityState.DEAD
        ]
        for eid in dead_ids:
            self.remove(eid)
        return len(dead_ids)
