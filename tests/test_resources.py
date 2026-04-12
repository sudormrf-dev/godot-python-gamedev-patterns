"""Tests for resource loading and management patterns."""

from __future__ import annotations

import pytest

from patterns.resources import (
    CachePolicy,
    LoadState,
    Resource,
    ResourceGroup,
    ResourceLoader,
    ResourcePath,
    ResourcePreloader,
    ResourceType,
)


class TestResourceType:
    def test_is_binary(self):
        assert ResourceType.TEXTURE.is_binary()
        assert ResourceType.AUDIO.is_binary()
        assert ResourceType.MESH.is_binary()
        assert ResourceType.FONT.is_binary()
        assert not ResourceType.SCRIPT.is_binary()
        assert not ResourceType.SCENE.is_binary()

    def test_is_gpu_resource(self):
        assert ResourceType.TEXTURE.is_gpu_resource()
        assert ResourceType.SHADER.is_gpu_resource()
        assert ResourceType.MESH.is_gpu_resource()
        assert ResourceType.MATERIAL.is_gpu_resource()
        assert not ResourceType.AUDIO.is_gpu_resource()
        assert not ResourceType.SCRIPT.is_gpu_resource()


class TestLoadState:
    def test_is_available(self):
        assert LoadState.LOADED.is_available()
        assert not LoadState.LOADING.is_available()
        assert not LoadState.FAILED.is_available()

    def test_is_terminal(self):
        assert LoadState.LOADED.is_terminal()
        assert LoadState.FAILED.is_terminal()
        assert not LoadState.LOADING.is_terminal()
        assert not LoadState.NOT_LOADED.is_terminal()


class TestCachePolicy:
    def test_should_cache(self):
        assert CachePolicy.SESSION.should_cache()
        assert CachePolicy.PERMANENT.should_cache()
        assert CachePolicy.LRU.should_cache()
        assert not CachePolicy.NEVER.should_cache()


class TestResourcePath:
    def test_uid(self):
        rp1 = ResourcePath("res://player.png", ResourceType.TEXTURE)
        rp2 = ResourcePath("res://enemy.png", ResourceType.TEXTURE)
        assert rp1.uid() != rp2.uid()
        assert len(rp1.uid()) == 16

    def test_extension(self):
        rp = ResourcePath("res://player.png", ResourceType.TEXTURE)
        assert rp.extension() == "png"

    def test_extension_no_ext(self):
        rp = ResourcePath("res://player", ResourceType.TEXTURE)
        assert rp.extension() == ""

    def test_is_gdres(self):
        assert ResourcePath("data.tres", ResourceType.SCENE).is_gdres()
        assert ResourcePath("data.res", ResourceType.MATERIAL).is_gdres()
        assert not ResourcePath("data.png", ResourceType.TEXTURE).is_gdres()

    def test_filename(self):
        rp = ResourcePath("res://assets/player.png", ResourceType.TEXTURE)
        assert rp.filename() == "player.png"

    def test_directory(self):
        rp = ResourcePath("res://assets/player.png", ResourceType.TEXTURE)
        assert rp.directory() == "res://assets"


class TestResource:
    def test_ref_counting(self):
        rp = ResourcePath("res://x.png", ResourceType.TEXTURE)
        res = Resource(path=rp)
        res.acquire()
        res.acquire()
        assert res.ref_count == 2
        assert res.is_referenced()
        res.release()
        assert res.ref_count == 1
        res.release()
        assert not res.is_referenced()

    def test_release_clamps_at_zero(self):
        rp = ResourcePath("res://x.png", ResourceType.TEXTURE)
        res = Resource(path=rp)
        res.release()
        assert res.ref_count == 0

    def test_mark_loaded(self):
        rp = ResourcePath("res://x.png", ResourceType.TEXTURE)
        res = Resource(path=rp)
        res.mark_loaded(1024)
        assert res.is_loaded()
        assert res.size_bytes == 1024

    def test_mark_failed(self):
        rp = ResourcePath("res://x.png", ResourceType.TEXTURE)
        res = Resource(path=rp)
        res.mark_failed()
        assert res.state == LoadState.FAILED
        assert not res.is_loaded()

    def test_size_mb(self):
        rp = ResourcePath("res://x.png", ResourceType.TEXTURE)
        res = Resource(path=rp, size_bytes=1024 * 1024)
        assert res.size_mb() == 1.0

    def test_should_evict_lru(self):
        rp = ResourcePath("res://x.png", ResourceType.TEXTURE)
        res = Resource(path=rp, cache_policy=CachePolicy.LRU)
        res.mark_loaded()
        assert res.should_evict()
        res.acquire()
        assert not res.should_evict()


class TestResourceLoader:
    def test_load_and_cache(self):
        loader = ResourceLoader()
        res = loader.load("res://player.png", ResourceType.TEXTURE)
        assert res.state == LoadState.LOADING
        assert loader.loaded_count() == 0
        res.mark_loaded(512)
        assert loader.loaded_count() == 1

    def test_load_same_path_twice(self):
        loader = ResourceLoader()
        res1 = loader.load("res://player.png", ResourceType.TEXTURE)
        res2 = loader.load("res://player.png", ResourceType.TEXTURE)
        assert res1 is res2

    def test_unload(self):
        loader = ResourceLoader()
        res = loader.load("res://player.png", ResourceType.TEXTURE)
        res.mark_loaded(512)
        result = loader.unload("res://player.png")
        assert result
        assert loader.get("res://player.png") is None

    def test_unload_permanent(self):
        loader = ResourceLoader()
        res = loader.load("res://core.tres", ResourceType.SCENE)
        res.cache_policy = CachePolicy.PERMANENT
        res.mark_loaded(256)
        result = loader.unload("res://core.tres")
        assert not result

    def test_evict_lru(self):
        loader = ResourceLoader()
        for i in range(3):
            res = loader.load(f"res://texture_{i}.png", ResourceType.TEXTURE)
            res.cache_policy = CachePolicy.LRU
            res.mark_loaded(1024 * 1024)
        evicted = loader.evict_lru(1.5)
        assert evicted > 0

    def test_get(self):
        loader = ResourceLoader()
        loader.load("res://x.png", ResourceType.TEXTURE)
        assert loader.get("res://x.png") is not None
        assert loader.get("res://missing.png") is None

    def test_cache_size_mb(self):
        loader = ResourceLoader()
        res = loader.load("res://big.png", ResourceType.TEXTURE)
        res.mark_loaded(2 * 1024 * 1024)
        assert loader.cache_size_mb() == 2.0


class TestResourceGroup:
    def test_add_and_count(self):
        group = ResourceGroup(name="level1")
        group.add("res://player.png", ResourceType.TEXTURE)
        group.add("res://bgm.ogg", ResourceType.AUDIO)
        assert group.count() == 2

    def test_by_type(self):
        group = ResourceGroup(name="level1")
        group.add("res://player.png", ResourceType.TEXTURE)
        group.add("res://enemy.png", ResourceType.TEXTURE)
        group.add("res://bgm.ogg", ResourceType.AUDIO)
        textures = group.by_type(ResourceType.TEXTURE)
        assert len(textures) == 2

    def test_gpu_resources(self):
        group = ResourceGroup(name="graphics")
        group.add("res://sprite.png", ResourceType.TEXTURE)
        group.add("res://shader.gdshader", ResourceType.SHADER)
        group.add("res://sound.ogg", ResourceType.AUDIO)
        gpu = group.gpu_resources()
        assert len(gpu) == 2


class TestResourcePreloader:
    def test_preload_group(self):
        loader = ResourceLoader()
        preloader = ResourcePreloader(loader)
        group = ResourceGroup(name="ui")
        group.add("res://ui/button.png", ResourceType.TEXTURE)
        group.add("res://ui/font.ttf", ResourceType.FONT)
        preloader.register_group(group)
        loaded = preloader.preload_group("ui")
        assert loaded == 2

    def test_preload_missing_group(self):
        loader = ResourceLoader()
        preloader = ResourcePreloader(loader)
        assert preloader.preload_group("missing") == 0

    def test_group_count(self):
        loader = ResourceLoader()
        preloader = ResourcePreloader(loader)
        preloader.register_group(ResourceGroup(name="a"))
        preloader.register_group(ResourceGroup(name="b"))
        assert preloader.group_count() == 2


@pytest.mark.asyncio
async def test_resource_loader_async():
    loader = ResourceLoader()
    res = loader.load("res://async_test.png", ResourceType.TEXTURE)
    res.mark_loaded(4096)
    assert res.is_loaded()
    assert loader.loaded_count() == 1
