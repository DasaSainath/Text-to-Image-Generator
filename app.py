import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Load the model
@st.cache_resource  # Cache the model so it doesn't reload every time
def load_model():
    model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

pipe = load_model()

# Streamlit UI
st.title("🖼️ Text-to-Image Generator 🚀")
prompt = st.text_input("Enter a text prompt below to generate an image:")

if st.button("Generate Image"):
    with st.spinner(f"Generating an image for: {prompt}..."):
        image = pipe(prompt).images[0]  # Generate image
        image.save("generated_image.png")  # Save image
        st.image("generated_image.png")  # Display in Streamlit
