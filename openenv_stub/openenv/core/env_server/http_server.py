"""Stub for openenv.core.env_server.http_server."""
from typing import Any, Callable, Optional, Type
from openenv.core.env_server.types import Action, Observation


def create_app(env, action_cls, observation_cls, env_name=None, max_concurrent_envs=1, **kwargs):
    """Stub — returns None when FastAPI is not available."""
    try:
        from fastapi import FastAPI
        app = FastAPI(title=env_name or "SupportTicketEnv")
        return app
    except ImportError:
        return None
