import gradio as gr
import torch
import requests
from io import BytesIO
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from utils import hi 

def greet(name):
    # return "Hello " + name + "!!"
    return hi() +""+ name

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()