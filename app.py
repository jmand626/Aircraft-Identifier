import os
os.environ.pop("SSL_CERT_FILE", None) # Fix for Windows SSL error

import gradio as gr
import torch
from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

with open("class_names.txt", "r") as f:
    class_names = [name.strip() for name in f.readlines()]

model, transforms = create_effnetb2_model(num_classes=len(class_names))
model.load_state_dict(
    torch.load("aircraft_model.pth", map_location=torch.device("cpu"))
)

def predict(img) -> Tuple[Dict, float]:
    start_time = timer()
    img = transforms(img).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img), dim=1)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    pred_time = round(timer() - start_time, 5)
    return pred_labels_and_probs, pred_time

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=3, label="Top 3 Predictions"), gr.Number(label="Time (s)")],
    examples=[["examples/" + e] for e in os.listdir("examples")],
    title="FGVC Aircraft Identifier ✈️",
    description="Identifies 100 aircraft variants.",
    cache_examples=False
)
demo.launch()
