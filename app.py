import streamlit as st

st.title("Text-to-Image Generator 🚀")
st.write("Enter a text prompt below to generate an image.")

prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image"):
    st.write(f"Generating an image for: {prompt}")
    # Here, you would call your Stable Diffusion model
