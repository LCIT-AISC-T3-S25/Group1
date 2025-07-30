
import streamlit as st
import base64

# ------------------ Page Config ------------------ #
st.set_page_config(page_title="Explore AI", page_icon="🤖", layout="wide")

# ------------------ Background Image Function ------------------ #
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

# Call the function with your background image filename
add_bg_from_local("bg_img.jpg")

# ------------------ Load External CSS ------------------ #
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------ Header ------------------ #
st.markdown("<div class='main-title'>Explore AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>An immersive experience where you can delve into and engage with groundbreaking technologies across two premier domains of Artificial Intelligence (AI)</div>", unsafe_allow_html=True)
st.divider()

# ------------------ Layout Columns ------------------ #
col1, col2 = st.columns(2)

# ------------------ NLP Section ------------------ #
with col1:
    st.markdown("<div class='section-title'>NATURAL LANGUAGE PROCESSING</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>AI to understand, interpret, and generate human language</div>", unsafe_allow_html=True)

    # Tweet Sentiment Analysis Card
    st.markdown("""
    <div class="model-card">
        <div class="model-title">Tweet Sentiment Analysis</div>
        <div class="model-desc">Transformer-based model that classifies text into sentiment with interpretability features</div>
        <ul>
            <li>Transformer-based architecture</li>
            <li>Key word highlighting</li>
            <li>Three-class classification (Positive, Negative, or Neutral)</li>
            <li>Interpretable predictions</li>
        </ul>
        <form action="/Sentiment_Analysis_(Transformer)" target="_self">
            <button type="submit">Try It</button>
        </form>
    </div>
    """, unsafe_allow_html=True)

    # AI Chatbot Card
    st.markdown("""
    <div class="model-card">
        <div class="model-title">Chatbot (AI)</div>
        <div class="model-desc">A RAG based chatbot that retrieves relevant context from a biomedical knowledge base and provides accurate answers</div>
        <ul>
            <li>Knowledge Base Retrieval (Biomedical)</li>
            <li>Context-Aware Responses</li>
            <li>Dynamic Answer Generation</li>
            <li>Improved Accuracy</li>
        </ul>
        <form action="/AI_Chatbot_(RAG)" target="_self">
            <button type="submit">Try It</button>
        </form>
    </div>
    """, unsafe_allow_html=True)

# ------------------ CV Section ------------------ #
with col2:
    st.markdown("<div class='section-title'>COMPUTER VISION</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-subtitle'>AI to analyze, interpret, and generate visual data.</div>", unsafe_allow_html=True)

    # Image Generation Card
    st.markdown("""
    <div class="model-card">
        <div class="model-title">Image Generation</div>
        <div class="model-desc">A GAN model capable of creating realistic synthetic images</div>
        <ul>
            <li>Adversarial Training</li>
            <li>Random Noise/Conditional Input</li>
            <li>High-Quality Output</li>
            <li>Diverse Applications</li>
        </ul>
        <form action="/Image_Generation_(GAN)" target="_self">
            <button type="submit">Try It</button>
        </form>
    </div>
    """, unsafe_allow_html=True)

    # Text-to-Image Generation Card
    st.markdown("""
    <div class="model-card">
        <div class="model-title">Text-to-Image Generation</div>
        <div class="model-desc">A diffusion-based (GLIDE) model that generates high-quality, photo-realistic images from textual prompts and descriptions</div>
        <ul>
            <li>Caption-Guided Image Generation</li>
            <li>Diffusion-Based Refinement</li>
            <li>High-Res. Output</li>
            <li>Flexible Prompting</li>
        </ul>
        <form action="/Text_to_Image_Generation_(GLIDE)" target="_self">
            <button type="submit">Try It</button>
        </form>
    </div>
    """, unsafe_allow_html=True)
