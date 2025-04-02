import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.title("🖼️ Text-to-Image Generator 🚀")
st.write("Enter a text prompt below to generate an image.")

@st.cache_resource
def load_model():
    model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    model.to("cpu")  # Force CPU mode
    return model

pipe = load_model()
prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image"):
    with st.spinner(f"Generating an image for: {prompt}..."):
        image = pipe(prompt).images[0]  # Generate image
        st.image(image, caption="Generated Image", use_column_width=True)


