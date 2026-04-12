"""Tests for entity-component patterns."""

from __future__ import annotations

import pytest

from patterns.entities import (
    ComponentType,
    Entity,
    EntityRegistry,
    EntityState,
    EntityTag,
    HealthComponent,
    MovementComponent,
    TransformComponent,
    Vector2,
)


class TestVector2:
    def test_length_zero(self):
        v = Vector2(0.0, 0.0)
        assert v.length() == 0.0
        assert v.is_zero()

    def test_length(self):
        v = Vector2(3.0, 4.0)
        assert v.length() == 5.0

    def test_distance_to(self):
        a = Vector2(0.0, 0.0)
        b = Vector2(3.0, 4.0)
        assert a.distance_to(b) == 5.0

    def test_add(self):
        a = Vector2(1.0, 2.0)
        b = Vector2(3.0, 4.0)
        result = a.add(b)
        assert result.x == 4.0
        assert result.y == 6.0

    def test_scale(self):
        v = Vector2(2.0, 3.0)
        result = v.scale(2.0)
        assert result.x == 4.0
        assert result.y == 6.0


class TestComponentType:
    def test_is_physics(self):
        assert ComponentType.MOVEMENT.is_physics()
        assert ComponentType.COLLISION.is_physics()
        assert not ComponentType.HEALTH.is_physics()
        assert not ComponentType.SPRITE.is_physics()

    def test_is_visual(self):
        assert ComponentType.SPRITE.is_visual()
        assert ComponentType.ANIMATION.is_visual()
        assert not ComponentType.HEALTH.is_visual()


class TestEntityState:
    def test_is_alive(self):
        assert EntityState.IDLE.is_alive()
        assert EntityState.ACTIVE.is_alive()
        assert EntityState.SPAWNING.is_alive()
        assert not EntityState.DEAD.is_alive()
        assert not EntityState.DYING.is_alive()

    def test_can_interact(self):
        assert EntityState.ACTIVE.can_interact()
        assert not EntityState.IDLE.can_interact()
        assert not EntityState.DEAD.can_interact()


class TestEntityTag:
    def test_is_combatant(self):
        assert EntityTag.PLAYER.is_combatant()
        assert EntityTag.ENEMY.is_combatant()
        assert EntityTag.BOSS.is_combatant()
        assert not EntityTag.PICKUP.is_combatant()

    def test_is_collectible(self):
        assert EntityTag.PICKUP.is_collectible()
        assert not EntityTag.PLAYER.is_collectible()


class TestTransformComponent:
    def test_translate(self):
        transform = TransformComponent()
        transform.translate(Vector2(10.0, 5.0))
        assert transform.position.x == 10.0
        assert transform.position.y == 5.0

    def test_rotate(self):
        transform = TransformComponent()
        transform.rotate(1.5)
        assert transform.rotation == 1.5

    def test_flip(self):
        transform = TransformComponent(scale=Vector2(-1.0, 1.0))
        assert transform.is_flipped_h()
        assert not transform.is_flipped_v()


class TestHealthComponent:
    def test_initial_hp(self):
        health = HealthComponent(max_hp=100.0)
        assert health.current_hp == 100.0
        assert health.is_full_hp()

    def test_take_damage(self):
        health = HealthComponent(max_hp=100.0)
        actual = health.take_damage(30.0)
        assert actual == 30.0
        assert health.current_hp == 70.0
        assert not health.is_full_hp()

    def test_armor_reduction(self):
        health = HealthComponent(max_hp=100.0, armor=10.0)
        actual = health.take_damage(30.0)
        assert actual == 20.0
        assert health.current_hp == 80.0

    def test_damage_capped_at_zero(self):
        health = HealthComponent(max_hp=100.0)
        health.take_damage(200.0)
        assert health.current_hp == 0.0
        assert not health.is_alive()

    def test_heal(self):
        health = HealthComponent(max_hp=100.0)
        health.take_damage(50.0)
        healed = health.heal(30.0)
        assert healed == 30.0
        assert health.current_hp == 80.0

    def test_heal_capped_at_max(self):
        health = HealthComponent(max_hp=100.0)
        health.take_damage(10.0)
        healed = health.heal(50.0)
        assert healed == 10.0
        assert health.current_hp == 100.0

    def test_hp_percent(self):
        health = HealthComponent(max_hp=200.0, current_hp=100.0)
        assert health.hp_percent() == 0.5

    def test_hp_percent_zero_max(self):
        health = HealthComponent(max_hp=0.0, current_hp=0.0)
        assert health.hp_percent() == 0.0

    def test_tick_regen(self):
        health = HealthComponent(max_hp=100.0, regen_rate=10.0)
        health.take_damage(50.0)
        healed = health.tick_regen(1.0)
        assert healed == 10.0

    def test_tick_regen_no_regen_at_full(self):
        health = HealthComponent(max_hp=100.0, regen_rate=10.0)
        healed = health.tick_regen(1.0)
        assert healed == 0.0


class TestMovementComponent:
    def test_apply_force(self):
        mv = MovementComponent(max_speed=200.0)
        mv.apply_force(Vector2(50.0, 0.0))
        assert mv.velocity.x == 50.0
        assert mv.is_moving()

    def test_speed_capped(self):
        mv = MovementComponent(max_speed=100.0)
        mv.apply_force(Vector2(1000.0, 0.0))
        assert mv.velocity.length() <= 100.0

    def test_apply_friction(self):
        mv = MovementComponent(friction=0.5)
        mv.velocity = Vector2(100.0, 0.0)
        mv.apply_friction()
        assert mv.velocity.x == 50.0

    def test_speed_ratio(self):
        mv = MovementComponent(max_speed=100.0)
        mv.velocity = Vector2(50.0, 0.0)
        assert mv.speed_ratio() == 0.5

    def test_speed_ratio_zero_max(self):
        mv = MovementComponent(max_speed=0.0)
        assert mv.speed_ratio() == 0.0


class TestEntity:
    def test_initial_state(self):
        entity = Entity("e1", "Player")
        assert entity.entity_id == "e1"
        assert entity.name == "Player"
        assert entity.state == EntityState.SPAWNING

    def test_activate(self):
        entity = Entity("e1", "Player")
        assert entity.activate()
        assert entity.state == EntityState.ACTIVE
        assert not entity.activate()

    def test_kill(self):
        entity = Entity("e1", "Enemy")
        entity.activate()
        assert entity.kill()
        assert entity.state == EntityState.DYING
        assert not entity.kill()

    def test_destroy(self):
        entity = Entity("e1", "Enemy")
        entity.activate()
        entity.kill()
        entity.destroy()
        assert entity.state == EntityState.DEAD

    def test_tags(self):
        entity = Entity("e1", "Player")
        entity.add_tag(EntityTag.PLAYER)
        assert entity.has_tag(EntityTag.PLAYER)
        assert not entity.has_tag(EntityTag.ENEMY)

    def test_components(self):
        entity = Entity("e1", "Player")
        health = HealthComponent(max_hp=100.0)
        entity.add_component(ComponentType.HEALTH, health)
        assert entity.has_component(ComponentType.HEALTH)
        assert entity.get_component(ComponentType.HEALTH) is health
        assert not entity.has_component(ComponentType.MOVEMENT)
        assert entity.component_count() == 1

    def test_is_alive(self):
        entity = Entity("e1", "Player")
        assert entity.is_alive()
        entity.activate()
        entity.kill()
        entity.destroy()
        assert not entity.is_alive()


class TestEntityRegistry:
    def test_register_and_get(self):
        registry = EntityRegistry()
        entity = Entity("e1", "Player")
        entity.add_tag(EntityTag.PLAYER)
        registry.register(entity)
        assert registry.get("e1") is entity

    def test_get_by_tag(self):
        registry = EntityRegistry()
        p = Entity("p1", "Player")
        p.add_tag(EntityTag.PLAYER)
        e = Entity("e1", "Enemy")
        e.add_tag(EntityTag.ENEMY)
        registry.register(p)
        registry.register(e)
        players = registry.get_by_tag(EntityTag.PLAYER)
        assert len(players) == 1
        assert players[0].name == "Player"

    def test_remove(self):
        registry = EntityRegistry()
        entity = Entity("e1", "Enemy")
        registry.register(entity)
        assert registry.remove("e1")
        assert registry.get("e1") is None
        assert not registry.remove("e1")

    def test_alive_count(self):
        registry = EntityRegistry()
        e1 = Entity("e1", "A")
        e2 = Entity("e2", "B")
        e2.activate()
        e2.kill()
        e2.destroy()
        registry.register(e1)
        registry.register(e2)
        assert registry.alive_count() == 1

    def test_cleanup_dead(self):
        registry = EntityRegistry()
        alive = Entity("e1", "Alive")
        dead = Entity("e2", "Dead")
        dead.activate()
        dead.kill()
        dead.destroy()
        registry.register(alive)
        registry.register(dead)
        cleaned = registry.cleanup_dead()
        assert cleaned == 1
        assert registry.total_count() == 1

    def test_total_count(self):
        registry = EntityRegistry()
        for i in range(5):
            registry.register(Entity(f"e{i}", f"E{i}"))
        assert registry.total_count() == 5


@pytest.mark.asyncio
async def test_entity_full_lifecycle():
    registry = EntityRegistry()
    entity = Entity("hero", "Hero")
    entity.add_tag(EntityTag.PLAYER)
    health = HealthComponent(max_hp=100.0)
    entity.add_component(ComponentType.HEALTH, health)
    registry.register(entity)
    entity.activate()
    assert entity.state == EntityState.ACTIVE
    comp = entity.get_component(ComponentType.HEALTH)
    assert comp is not None
    comp.take_damage(100.0)
    assert not comp.is_alive()
    entity.kill()
    entity.destroy()
    registry.cleanup_dead()
    assert registry.total_count() == 0
