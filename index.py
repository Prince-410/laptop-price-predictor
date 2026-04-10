from fastapi import FastAPI
import gradio as gr
from app import demo # Import the demo from our main file

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Laptop Price Predictor API is running. Go to /gradio for the interface."}

# Mount the Gradio app to the root
app = gr.mount_gradio_app(app, demo, path="/")
