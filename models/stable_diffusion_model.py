import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.title("🖼️ Text-to-Image Generator 🚀")
st.write("Enter a text prompt below to generate an image.")

# Load the model
@st.cache_resource  # Cache the model so it doesn't reload every time
def load_model():
    model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
    return model

# Initialize the pipeline
pipe = load_model()

# Streamlit UI
prompt = st.text_input("Enter a text prompt below to generate an image:")

if st.button("Generate Image"):
    with st.spinner(f"Generating an image for: {prompt}..."):
        image = pipe(prompt).images[0]  # Generate image
        st.image(image, caption="Generated Image", use_column_width=True)
        image.save("generated_image.png")  # Save image
        st.image("generated_image.png")  # Display in Streamlit
