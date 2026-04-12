"""Tests for scene tree and node management patterns."""

from __future__ import annotations

import pytest

from patterns.scene import (
    NodePath,
    NodeType,
    Scene,
    SceneManager,
    SceneNode,
    SceneState,
)


class TestNodeType:
    def test_is_physics(self):
        assert NodeType.RIGID_BODY.is_physics()
        assert NodeType.CHARACTER_BODY.is_physics()
        assert NodeType.STATIC_BODY.is_physics()
        assert not NodeType.SPRITE.is_physics()
        assert not NodeType.LABEL.is_physics()

    def test_is_ui(self):
        assert NodeType.CONTROL.is_ui()
        assert NodeType.LABEL.is_ui()
        assert NodeType.BUTTON.is_ui()
        assert not NodeType.NODE2D.is_ui()

    def test_is_2d(self):
        assert NodeType.NODE2D.is_2d()
        assert NodeType.SPRITE.is_2d()
        assert NodeType.AREA2D.is_2d()
        assert not NodeType.NODE3D.is_2d()


class TestSceneState:
    def test_is_active(self):
        assert SceneState.READY.is_active()
        assert SceneState.PAUSED.is_active()
        assert not SceneState.UNLOADED.is_active()
        assert not SceneState.STOPPING.is_active()

    def test_can_transition(self):
        assert SceneState.UNLOADED.can_transition_to(SceneState.LOADING)
        assert not SceneState.UNLOADED.can_transition_to(SceneState.READY)
        assert SceneState.LOADING.can_transition_to(SceneState.READY)
        assert SceneState.LOADING.can_transition_to(SceneState.UNLOADED)
        assert SceneState.READY.can_transition_to(SceneState.PAUSED)
        assert SceneState.READY.can_transition_to(SceneState.STOPPING)
        assert not SceneState.READY.can_transition_to(SceneState.LOADING)
        assert SceneState.PAUSED.can_transition_to(SceneState.READY)
        assert SceneState.STOPPING.can_transition_to(SceneState.UNLOADED)


class TestNodePath:
    def test_from_string_root(self):
        path = NodePath.from_string("/")
        assert path.depth() == 0
        assert str(path) == "/"

    def test_from_string_segments(self):
        path = NodePath.from_string("/Root/Player/Sprite")
        assert path.depth() == 3
        assert path.name() == "Sprite"

    def test_parent(self):
        path = NodePath.from_string("/Root/Player/Sprite")
        parent = path.parent()
        assert parent.name() == "Player"
        assert parent.depth() == 2

    def test_child(self):
        path = NodePath.from_string("/Root/Player")
        child = path.child("Weapon")
        assert str(child) == "/Root/Player/Weapon"

    def test_empty_path(self):
        path = NodePath.from_string("")
        assert path.depth() == 0
        assert path.name() == ""

    def test_is_absolute(self):
        path = NodePath.from_string("/Root")
        assert path.is_absolute()


class TestSceneNode:
    def test_add_remove_child(self):
        parent = SceneNode(name="Parent")
        child = SceneNode(name="Child")
        parent.add_child(child)
        assert parent.child_count() == 1
        removed = parent.remove_child("Child")
        assert removed is not None
        assert removed.name == "Child"
        assert parent.child_count() == 0

    def test_get_child(self):
        parent = SceneNode(name="Parent")
        child = SceneNode(name="Target")
        parent.add_child(child)
        assert parent.get_child("Target") is child
        assert parent.get_child("Missing") is None

    def test_groups(self):
        node = SceneNode(name="Node")
        node.add_to_group("enemies")
        assert node.is_in_group("enemies")
        assert not node.is_in_group("players")

    def test_properties(self):
        node = SceneNode(name="Node")
        node.set_property("speed", 100)
        assert node.get_property("speed") == 100
        assert node.get_property("missing", "default") == "default"

    def test_all_descendants(self):
        root = SceneNode(name="Root")
        child1 = SceneNode(name="Child1")
        child2 = SceneNode(name="Child2")
        grandchild = SceneNode(name="Grandchild")
        child1.add_child(grandchild)
        root.add_child(child1)
        root.add_child(child2)
        descendants = root.all_descendants()
        assert len(descendants) == 3

    def test_find_by_type(self):
        root = SceneNode(name="Root")
        sprite = SceneNode(name="Sprite", node_type=NodeType.SPRITE)
        label = SceneNode(name="Label", node_type=NodeType.LABEL)
        root.add_child(sprite)
        root.add_child(label)
        sprites = root.find_by_type(NodeType.SPRITE)
        assert len(sprites) == 1
        assert sprites[0].name == "Sprite"

    def test_find_in_group(self):
        root = SceneNode(name="Root")
        enemy1 = SceneNode(name="Enemy1")
        enemy2 = SceneNode(name="Enemy2")
        enemy1.add_to_group("enemies")
        enemy2.add_to_group("enemies")
        root.add_child(enemy1)
        root.add_child(enemy2)
        enemies = root.find_in_group("enemies")
        assert len(enemies) == 2


class TestScene:
    def test_transition_valid(self):
        root = SceneNode(name="Root")
        scene = Scene(name="Main", root=root)
        assert scene.transition_to(SceneState.LOADING)
        assert scene.state == SceneState.LOADING

    def test_transition_invalid(self):
        root = SceneNode(name="Root")
        scene = Scene(name="Main", root=root)
        assert not scene.transition_to(SceneState.READY)
        assert scene.state == SceneState.UNLOADED

    def test_is_ready(self):
        root = SceneNode(name="Root")
        scene = Scene(name="Main", root=root, state=SceneState.READY)
        assert scene.is_ready()

    def test_total_node_count(self):
        root = SceneNode(name="Root")
        root.add_child(SceneNode(name="Child"))
        scene = Scene(name="Main", root=root)
        assert scene.total_node_count() == 2

    def test_get_nodes_in_group(self):
        root = SceneNode(name="Root")
        enemy = SceneNode(name="Enemy")
        enemy.add_to_group("enemies")
        root.add_child(enemy)
        scene = Scene(name="Main", root=root)
        assert len(scene.get_nodes_in_group("enemies")) == 1
        assert len(scene.get_nodes_in_group("players")) == 0


class TestSceneManager:
    def test_register_and_get(self):
        manager = SceneManager()
        root = SceneNode(name="Root")
        scene = Scene(name="Main", root=root)
        manager.register(scene)
        assert manager.scene_count() == 1
        assert manager.get_scene("Main") is scene

    def test_load_and_activate(self):
        manager = SceneManager()
        root = SceneNode(name="Root")
        scene = Scene(name="Main", root=root)
        manager.register(scene)
        assert manager.load_scene("Main")
        assert manager.activate_scene("Main")
        active = manager.get_active()
        assert active is not None
        assert active.name == "Main"

    def test_load_unknown_scene(self):
        manager = SceneManager()
        assert not manager.load_scene("Unknown")

    def test_activate_without_load(self):
        manager = SceneManager()
        root = SceneNode(name="Root")
        scene = Scene(name="Main", root=root)
        manager.register(scene)
        assert not manager.activate_scene("Main")
        assert manager.get_active() is None

    def test_no_active_initially(self):
        manager = SceneManager()
        assert manager.get_active() is None


@pytest.mark.asyncio
async def test_scene_async_flow():
    manager = SceneManager()
    root = SceneNode(name="Root")
    scene = Scene(name="Level1", root=root)
    manager.register(scene)
    assert manager.load_scene("Level1")
    assert manager.activate_scene("Level1")
    active = manager.get_active()
    assert active is not None
    assert active.is_ready()
