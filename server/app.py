"""
FastAPI application entry point for the Support Ticket Environment.
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

# Mount our custom Gradio UI at /playground
try:
    import gradio as gr
    from support_ticket_env.gradio_ui import demo
    app = gr.mount_gradio_app(app, demo, path="/playground")
    print("Custom Gradio UI mounted at /playground")
except Exception as e:
    print(f"Gradio UI not mounted: {e}")