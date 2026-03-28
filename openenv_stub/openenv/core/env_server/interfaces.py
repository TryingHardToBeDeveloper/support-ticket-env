"""Stub for openenv.core.env_server.interfaces."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
from openenv.core.env_server.types import Action, Observation, State


class Environment(ABC):
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self, transform=None, rubric=None):
        self.rubric = rubric

    @abstractmethod
    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> Observation:
        ...

    @abstractmethod
    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs) -> Observation:
        ...

    @property
    @abstractmethod
    def state(self) -> State:
        ...

    def close(self) -> None:
        pass
