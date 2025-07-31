import streamlit as st
import base64
from config import ENDPOINTS, PAGES, APP_INFO

# ------------------ Page Config ------------------ #
st.set_page_config(page_title="AI Model Selector", page_icon="🤖", layout="wide")

# ------------------ Background Image ------------------ #
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    bg_css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

add_bg_from_local("bg_img.jpg")

# ------------------ External CSS ------------------ #
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------ Header ------------------ #
st.markdown(f"<div class='main-title'>{APP_INFO['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-title'>{APP_INFO['subtitle']}</div>", unsafe_allow_html=True)
st.divider()

# ------------------ Layout ------------------ #
col1, col2 = st.columns(2)

# ----------- NLP Section ----------- #
with col1:
    st.markdown("<div class='section-title'>NATURAL LANGUAGE PROCESSING</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>AI that understands and generates language</div>", unsafe_allow_html=True)

    # Sentiment Transformer
    model = ENDPOINTS['sentiment']
    st.markdown(f"""
    <div class="model-card">
        <div class="model-title">{model['name']}</div>
        <div class="model-desc">{model['description']}</div>
        <ul>
            <li>3-Class Sentiment</li>
            <li>Interpretable predictions</li>
            <li>Highlighting key words</li>
        </ul>
        <form>
            <button type="submit">Launch</button>
        </form>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🧵 Launch Sentiment Transformer"):
        st.switch_page(PAGES[model["name"]])

    # RAG Chatbot
    model = ENDPOINTS['rag']
    st.markdown(f"""
    <div class="model-card">
        <div class="model-title">{model['name']}</div>
        <div class="model-desc">{model['description']}</div>
        <ul>
            <li>Biomedical knowledge base</li>
            <li>Context-aware answers</li>
        </ul>
        <form>
            <button type="submit">Launch</button>
        </form>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🪖 Launch RAG Medical Chatbot"):
        st.switch_page(PAGES[model["name"]])

# ----------- CV Section ----------- #
with col2:
    st.markdown("<div class='section-title'>COMPUTER VISION</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>AI that generates and understands images</div>", unsafe_allow_html=True)

    # CGAN Generator
    model = ENDPOINTS['gan']
    st.markdown(f"""
    <div class="model-card">
        <div class="model-title">{model['name']}</div>
        <div class="model-desc">{model['description']}</div>
        <ul>
            <li>Adversarial Training</li>
            <li>Conditioned on labels</li>
        </ul>
        <form>
            <button type="submit">Launch</button>
        </form>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🍜 Launch CGAN Generator"):
        st.switch_page(PAGES[model["name"]])

    # GLIDE Diffusion
    model = ENDPOINTS['glide']
    st.markdown(f"""
    <div class="model-card">
        <div class="model-title">{model['name']}</div>
        <div class="model-desc">{model['description']}</div>
        <ul>
            <li>Flexible prompt input</li>
            <li>Diffusion-based refinement</li>
        </ul>
        <form>
            <button type="submit">Launch</button>
        </form>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🎨 Launch GLIDE Diffusion"):
        st.switch_page(PAGES[model["name"]])