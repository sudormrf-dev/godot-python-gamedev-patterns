"""Godot resource loading and management patterns."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResourceType(str, Enum):
    TEXTURE = "texture"
    AUDIO = "audio"
    SCENE = "scene"
    SCRIPT = "script"
    SHADER = "shader"
    FONT = "font"
    MESH = "mesh"
    MATERIAL = "material"
    ANIMATION = "animation"
    TRANSLATION = "translation"

    def is_binary(self) -> bool:
        return self in {
            ResourceType.TEXTURE,
            ResourceType.AUDIO,
            ResourceType.MESH,
            ResourceType.FONT,
        }

    def is_gpu_resource(self) -> bool:
        return self in {
            ResourceType.TEXTURE,
            ResourceType.SHADER,
            ResourceType.MESH,
            ResourceType.MATERIAL,
        }


class LoadState(str, Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    UNLOADING = "unloading"

    def is_available(self) -> bool:
        return self == LoadState.LOADED

    def is_terminal(self) -> bool:
        return self in {LoadState.LOADED, LoadState.FAILED}


class CachePolicy(str, Enum):
    NEVER = "never"
    SESSION = "session"
    PERMANENT = "permanent"
    LRU = "lru"

    def should_cache(self) -> bool:
        return self != CachePolicy.NEVER


@dataclass
class ResourcePath:
    path: str
    resource_type: ResourceType

    def uid(self) -> str:
        return hashlib.sha256(self.path.encode()).hexdigest()[:16]

    def extension(self) -> str:
        parts = self.path.rsplit(".", 1)
        return parts[1].lower() if len(parts) > 1 else ""

    def is_gdres(self) -> bool:
        return self.extension() in {"tres", "res"}

    def filename(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def directory(self) -> str:
        parts = self.path.rsplit("/", 1)
        return parts[0] if len(parts) > 1 else ""


@dataclass
class Resource:
    path: ResourcePath
    state: LoadState = LoadState.NOT_LOADED
    size_bytes: int = 0
    ref_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    cache_policy: CachePolicy = CachePolicy.SESSION

    def acquire(self) -> Resource:
        self.ref_count += 1
        return self

    def release(self) -> Resource:
        if self.ref_count > 0:
            self.ref_count -= 1
        return self

    def is_loaded(self) -> bool:
        return self.state.is_available()

    def is_referenced(self) -> bool:
        return self.ref_count > 0

    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)

    def mark_loaded(self, size_bytes: int = 0) -> Resource:
        self.state = LoadState.LOADED
        if size_bytes > 0:
            self.size_bytes = size_bytes
        return self

    def mark_failed(self) -> Resource:
        self.state = LoadState.FAILED
        return self

    def should_evict(self) -> bool:
        return not self.is_referenced() and self.cache_policy == CachePolicy.LRU


class ResourceLoader:
    def __init__(self) -> None:
        self._cache: dict[str, Resource] = {}
        self._load_order: list[str] = []
        self._max_cache_mb: float = 512.0

    def load(self, path: str, resource_type: ResourceType) -> Resource:
        uid = hashlib.sha256(path.encode()).hexdigest()[:16]
        if uid in self._cache:
            return self._cache[uid].acquire()
        res_path = ResourcePath(path=path, resource_type=resource_type)
        resource = Resource(path=res_path)
        resource.state = LoadState.LOADING
        self._cache[uid] = resource
        self._load_order.append(uid)
        return resource

    def unload(self, path: str) -> bool:
        uid = hashlib.sha256(path.encode()).hexdigest()[:16]
        resource = self._cache.get(uid)
        if resource is None:
            return False
        resource.release()
        if (
            not resource.is_referenced()
            and resource.cache_policy != CachePolicy.PERMANENT
        ):
            del self._cache[uid]
            if uid in self._load_order:
                self._load_order.remove(uid)
            return True
        return False

    def cache_size_mb(self) -> float:
        return sum(r.size_mb() for r in self._cache.values() if r.is_loaded())

    def evict_lru(self, target_mb: float) -> int:
        evicted = 0
        for uid in list(self._load_order):
            if self.cache_size_mb() <= target_mb:
                break
            resource = self._cache.get(uid)
            if resource and resource.should_evict():
                del self._cache[uid]
                self._load_order.remove(uid)
                evicted += 1
        return evicted

    def loaded_count(self) -> int:
        return sum(1 for r in self._cache.values() if r.is_loaded())

    def get(self, path: str) -> Resource | None:
        uid = hashlib.sha256(path.encode()).hexdigest()[:16]
        return self._cache.get(uid)


@dataclass
class ResourceGroup:
    name: str
    resources: list[ResourcePath] = field(default_factory=list)
    preload: bool = False

    def add(self, path: str, resource_type: ResourceType) -> ResourceGroup:
        self.resources.append(ResourcePath(path=path, resource_type=resource_type))
        return self

    def count(self) -> int:
        return len(self.resources)

    def by_type(self, resource_type: ResourceType) -> list[ResourcePath]:
        return [r for r in self.resources if r.resource_type == resource_type]

    def gpu_resources(self) -> list[ResourcePath]:
        return [r for r in self.resources if r.resource_type.is_gpu_resource()]


class ResourcePreloader:
    def __init__(self, loader: ResourceLoader) -> None:
        self._loader = loader
        self._groups: dict[str, ResourceGroup] = {}

    def register_group(self, group: ResourceGroup) -> ResourcePreloader:
        self._groups[group.name] = group
        return self

    def preload_group(self, name: str) -> int:
        group = self._groups.get(name)
        if group is None:
            return 0
        loaded = 0
        for res_path in group.resources:
            resource = self._loader.load(res_path.path, res_path.resource_type)
            resource.mark_loaded()
            loaded += 1
        return loaded

    def group_count(self) -> int:
        return len(self._groups)
