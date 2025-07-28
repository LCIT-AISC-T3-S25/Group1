import streamlit as st
import requests
from PIL import Image
import io

st.title("Conditional GAN Image Generator")

# Label options
label_map = {
    "Food": 0,
    "Drink": 1,
    "Inside": 2,
    "Outside": 3
}

# Dropdown for label selection
selected_label = st.selectbox("Select a label:", list(label_map.keys()))
label_id = label_map[selected_label]

# Generate button
if st.button("Generate Image"):
    try:
        # Send GET request with label as query param
        response = requests.get(f"http://localhost:8002/generate", params={"label": label_id})

        if response.status_code == 200:
            # Convert binary response to image
            image = Image.open(io.BytesIO(response.content))
            st.markdown(f"<h4 style='text-align: center;'>Generated for: {selected_label}</h4>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
        else:
            st.error(f"❌ Server responded with status code {response.status_code}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
