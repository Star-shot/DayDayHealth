import gradio as gr

from plot import *
from utils import *


def greet(name):
    return "Hello " + name + "!"

iface = gr.Interface(fn=greet, inputs=gr.Textbox(), outputs=gr.Textbox())
iface.launch(share=True)

