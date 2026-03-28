"""
Offline stub for openenv.core.env_server.types.
Uses stdlib only — no pydantic required.
"""
from __future__ import annotations
from typing import Any, Dict, Optional


class Action:
    def __init__(self, **kwargs):
        self.metadata = kwargs.pop("metadata", {})
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump(self):
        return {k: v for k, v in vars(self).items()}


class Observation:
    def __init__(self, **kwargs):
        self.done   = kwargs.pop("done", False)
        self.reward = kwargs.pop("reward", None)
        self.metadata = kwargs.pop("metadata", {})
        for k, v in kwargs.items():
            setattr(self, k, v)


class State:
    def __init__(self, **kwargs):
        self.episode_id = kwargs.pop("episode_id", None)
        self.step_count = kwargs.pop("step_count", 0)
        for k, v in kwargs.items():
            setattr(self, k, v)
