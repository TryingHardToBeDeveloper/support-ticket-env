"""Stub for openenv.core.env_client."""
from abc import ABC
from typing import Generic, TypeVar

ActT = TypeVar("ActT")
ObsT = TypeVar("ObsT")
StateT = TypeVar("StateT")


class EnvClient(ABC, Generic[ActT, ObsT, StateT]):
    def __init__(self, base_url: str, **kwargs):
        self.base_url = base_url
