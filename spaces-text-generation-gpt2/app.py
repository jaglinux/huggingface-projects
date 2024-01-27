import gradio as gr
from transformers import pipeline

def infer(input_text):
    out = pipeline("text-generation", model="gpt2")
    output_text = out(input_text, max_length=30, num_return_sequences=1)
    return output_text[0]['generated_text']

demo = gr.Interface(
    fn=infer,
    inputs=["text"],
    outputs=["text"],
    description="text-generation using gpt2",
)

demo.launch()
