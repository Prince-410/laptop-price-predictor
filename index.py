from fastapi import FastAPI
import gradio as gr

# Import the Gradio demo from your app.py
from app import demo

app = FastAPI()

# Mount the Gradio app to the root path
app = gr.mount_gradio_app(app, demo, path="/")
