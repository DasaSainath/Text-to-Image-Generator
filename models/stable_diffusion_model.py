from diffusers import StableDiffusionPipeline
import torch
from config import config

# Load the model
model = StableDiffusionPipeline.from_pretrained(config.MODEL_NAME, use_auth_token=config.HUGGING_FACE_API_KEY)
model.to("cuda")  # or "cpu" for CPU support

def generate_image_from_text(prompt):
    image = model(prompt).images[0]
    return image

