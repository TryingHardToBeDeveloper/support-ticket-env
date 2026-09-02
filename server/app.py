"""
FastAPI application entry point for the Support Ticket Environment.
"""

import os

from fastapi import Request
from fastapi.responses import JSONResponse
from openenv.core.env_server.http_server import create_app

from server.security import SecurityMiddleware
from server.support_environment import SupportTicketEnvironment
from support_ticket_env.models import SupportAction, SupportObservation

app = create_app(
    env=SupportTicketEnvironment,
    action_cls=SupportAction,
    observation_cls=SupportObservation,
    env_name="support_ticket_env",
    max_concurrent_envs=4,
)
app.add_middleware(SecurityMiddleware)


async def invalid_input_handler(request: Request, error: ValueError) -> JSONResponse:
    """Convert environment input errors into stable client responses."""
    return JSONResponse(status_code=422, content={"detail": str(error)})


app.add_exception_handler(ValueError, invalid_input_handler)

development = os.getenv("SUPPORT_ENV_MODE", "development").lower() != "production"
playground_enabled = os.getenv(
    "SUPPORT_ENV_ENABLE_PLAYGROUND", "true" if development else "false"
).lower() in {"1", "true", "yes", "on"}

if playground_enabled:
    try:
        import gradio as gr

        from gradio_ui import demo

        app = gr.mount_gradio_app(app, demo, path="/playground")
    except ImportError:
        # Gradio is an optional server extra; the API remains fully functional.
        pass


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
