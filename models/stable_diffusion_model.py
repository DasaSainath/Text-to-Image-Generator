import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.title("🖼️ Text-to-Image Generator 🚀")
st.write("Enter a text prompt below to generate an image.")

# ✅ Cache model loading
@st.cache_resource
def load_model():
    try:
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipe = load_model()

# ✅ Input prompt
prompt = st.text_input("Enter your prompt:")

if pipe and st.button("Generate Image"):
    with st.spinner(f"Generating an image for: {prompt}..."):
        try:
            image = pipe(prompt).images[0]  # Generate image
            st.image(image, caption="Generated Image", use_column_width=True)  # Display image
        except Exception as e:
            st.error(f"Error generating image: {e}")

