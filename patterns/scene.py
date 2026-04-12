"""Godot scene tree and node management patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    NODE = "Node"
    NODE2D = "Node2D"
    NODE3D = "Node3D"
    CONTROL = "Control"
    AREA2D = "Area2D"
    RIGID_BODY = "RigidBody2D"
    CHARACTER_BODY = "CharacterBody2D"
    STATIC_BODY = "StaticBody2D"
    SPRITE = "Sprite2D"
    LABEL = "Label"
    BUTTON = "Button"
    CAMERA = "Camera2D"

    def is_physics(self) -> bool:
        return self in {
            NodeType.RIGID_BODY,
            NodeType.CHARACTER_BODY,
            NodeType.STATIC_BODY,
        }

    def is_ui(self) -> bool:
        return self in {NodeType.CONTROL, NodeType.LABEL, NodeType.BUTTON}

    def is_2d(self) -> bool:
        return "2D" in self.value or self.value in {"Node2D", "Sprite2D"}


class SceneState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    PAUSED = "paused"
    STOPPING = "stopping"

    def is_active(self) -> bool:
        return self in {SceneState.READY, SceneState.PAUSED}

    def can_transition_to(self, target: SceneState) -> bool:
        transitions: dict[SceneState, set[SceneState]] = {
            SceneState.UNLOADED: {SceneState.LOADING},
            SceneState.LOADING: {SceneState.READY, SceneState.UNLOADED},
            SceneState.READY: {SceneState.PAUSED, SceneState.STOPPING},
            SceneState.PAUSED: {SceneState.READY, SceneState.STOPPING},
            SceneState.STOPPING: {SceneState.UNLOADED},
        }
        return target in transitions.get(self, set())


@dataclass
class NodePath:
    segments: list[str] = field(default_factory=list)

    @classmethod
    def from_string(cls, path: str) -> NodePath:
        if not path or path == "/":
            return cls(segments=[])
        return cls(segments=[s for s in path.strip("/").split("/") if s])

    def __str__(self) -> str:
        return "/" + "/".join(self.segments) if self.segments else "/"

    def is_absolute(self) -> bool:
        return True

    def depth(self) -> int:
        return len(self.segments)

    def parent(self) -> NodePath:
        if not self.segments:
            return NodePath(segments=[])
        return NodePath(segments=self.segments[:-1])

    def child(self, name: str) -> NodePath:
        return NodePath(segments=[*self.segments, name])

    def name(self) -> str:
        return self.segments[-1] if self.segments else ""


@dataclass
class SceneNode:
    name: str
    node_type: NodeType = NodeType.NODE
    children: list[SceneNode] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    groups: set[str] = field(default_factory=set)
    visible: bool = True
    process_mode: str = "inherit"

    def add_child(self, child: SceneNode) -> SceneNode:
        self.children.append(child)
        return self

    def remove_child(self, name: str) -> SceneNode | None:
        for i, child in enumerate(self.children):
            if child.name == name:
                return self.children.pop(i)
        return None

    def get_child(self, name: str) -> SceneNode | None:
        return next((c for c in self.children if c.name == name), None)

    def add_to_group(self, group: str) -> SceneNode:
        self.groups.add(group)
        return self

    def is_in_group(self, group: str) -> bool:
        return group in self.groups

    def child_count(self) -> int:
        return len(self.children)

    def set_property(self, key: str, value: Any) -> SceneNode:
        self.properties[key] = value
        return self

    def get_property(self, key: str, default: Any = None) -> Any:
        return self.properties.get(key, default)

    def all_descendants(self) -> list[SceneNode]:
        result: list[SceneNode] = []
        for child in self.children:
            result.append(child)
            result.extend(child.all_descendants())
        return result

    def find_by_type(self, node_type: NodeType) -> list[SceneNode]:
        return [n for n in self.all_descendants() if n.node_type == node_type]

    def find_in_group(self, group: str) -> list[SceneNode]:
        return [n for n in self.all_descendants() if n.is_in_group(group)]


@dataclass
class Scene:
    name: str
    root: SceneNode
    state: SceneState = SceneState.UNLOADED
    metadata: dict[str, Any] = field(default_factory=dict)

    def transition_to(self, state: SceneState) -> bool:
        if self.state.can_transition_to(state):
            self.state = state
            return True
        return False

    def get_nodes_in_group(self, group: str) -> list[SceneNode]:
        return self.root.find_in_group(group)

    def total_node_count(self) -> int:
        return 1 + len(self.root.all_descendants())

    def is_ready(self) -> bool:
        return self.state == SceneState.READY


class SceneManager:
    def __init__(self) -> None:
        self._scenes: dict[str, Scene] = {}
        self._active: str | None = None

    def register(self, scene: Scene) -> SceneManager:
        self._scenes[scene.name] = scene
        return self

    def load_scene(self, name: str) -> bool:
        if name not in self._scenes:
            return False
        scene = self._scenes[name]
        return scene.transition_to(SceneState.LOADING)

    def activate_scene(self, name: str) -> bool:
        if name not in self._scenes:
            return False
        scene = self._scenes[name]
        if scene.state == SceneState.LOADING and scene.transition_to(SceneState.READY):
            self._active = name
            return True
        return False

    def get_active(self) -> Scene | None:
        if self._active is None:
            return None
        return self._scenes.get(self._active)

    def scene_count(self) -> int:
        return len(self._scenes)

    def get_scene(self, name: str) -> Scene | None:
        return self._scenes.get(name)
