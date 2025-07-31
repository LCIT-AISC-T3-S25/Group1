import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
from config import ENDPOINTS

st.set_page_config(page_title="CGAN Image Generator", page_icon="🖼️")
st.title("🖼️ CGAN Image Generator")
st.caption("Select a category to generate realistic Yelp-style images using a conditional GAN.")

categories = ["food", "drink", "inside", "outside"]
category = st.selectbox("Choose an image category:", categories)

if st.button("Generate Image"):
    if not category:
        st.warning("Please select a category.")
    else:
        with st.spinner("Generating image..."):
            try:
                api_url = ENDPOINTS["gan"]["url"]
                response = requests.post(
                    api_url,
                    json={"category": category},
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") and "image" in data:
                        # Decode base64 and display the image
                        image_base64 = data["image"]
                        image_bytes = base64.b64decode(image_base64)
                        image = Image.open(BytesIO(image_bytes))
                        st.image(image, caption=f"Generated {category} image", use_column_width=True)
                    else:
                        st.error(f"Image generation failed: {data.get('error', 'Unknown error')}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")
