import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from config import ENDPOINTS

st.title("🖼️ GLIDE Diffusion - Text to Image Generator")

# Input caption
caption = st.text_input("Enter a caption for image generation:")

# Trigger generation
if st.button("Generate Image") and caption:
    with st.spinner("Generating image..."):
        try:
            api_url = ENDPOINTS["glide"]["url"]
            response = requests.post(
                api_url,
                json={"caption": caption},
                timeout=300
            )

            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                st.image(image, caption=caption)
            else:
                st.error(f"Failed to generate image: {response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")
