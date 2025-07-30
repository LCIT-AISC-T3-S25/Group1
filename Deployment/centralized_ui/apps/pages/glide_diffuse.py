import streamlit as st
import requests
import base64
from PIL import Image
import io

st.title(" GLIDE: Text-to-Image Generator")

caption = st.text_input("Enter a prompt (e.g. 'A cozy café in winter')")

if st.button("Generate Image") and caption:
    try:
        res = requests.post("http://localhost:8003/generate_from_caption", json={"caption": caption})
        img_data = res.json().get("image_base64")
        if img_data:
            image = Image.open(io.BytesIO(base64.b64decode(img_data)))
            st.image(image, caption=caption, use_container_width=True)
        else:
            st.warning("No image returned.")
    except Exception as e:
        st.error(f"Error: {e}")
