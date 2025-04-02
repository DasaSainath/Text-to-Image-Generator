import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

feature-branch
st.title("🖼️ Text-to-Image Generator 🚀")
st.write("Enter a text prompt below to generate an image.")

@st.cache_resource
def load_model():
    model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    model.to("cpu")  # Force CPU mode
    return model

pipe = load_model()
prompt = st.text_input("Enter your prompt:")
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
main

if st.button("Generate Image"):
    with st.spinner(f"Generating an image for: {prompt}..."):
        image = pipe(prompt).images[0]  # Generate image
feature-branch
        st.image(image, caption="Generated Image", use_column_width=True)



        image.save("generated_image.png")  # Save image
        st.image("generated_image.png")  # Display in Streamlit
main
