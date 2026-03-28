"""
FastAPI application entry point for the Support Ticket Environment.
Serves the OpenEnv HTTP/WebSocket API and optionally the Gradio UI at /web.
"""
import os
from openenv.core.env_server.http_server import create_app

from support_ticket_env.models import SupportAction, SupportObservation
from support_ticket_env.server.support_environment import SupportTicketEnvironment

app = create_app(
    env=SupportTicketEnvironment,
    action_cls=SupportAction,
    observation_cls=SupportObservation,
    env_name="support_ticket_env",
    max_concurrent_envs=4,
)

# Mount Gradio UI at /web when requested
if os.getenv("ENABLE_WEB_INTERFACE", "true").lower() == "true":
    try:
        import gradio as gr
        from support_ticket_env.gradio_ui import demo
        import gradio.routes
        app = gr.mount_gradio_app(app, demo, path="/web")
        print("Gradio UI mounted at /web")
    except Exception as e:
        print(f"Gradio UI not mounted: {e}")
