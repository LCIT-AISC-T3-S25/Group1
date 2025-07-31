# config.py

ENDPOINTS = {
    "sentiment": {
        "name": "Sentiment Transformer",
        "description": "Classifies tweets into Positive, Negative, or Neutral using a transformer model",
        "url": "http://localhost:8001"
    },
    "rag": {
        "name": "Medical Chatbot (RAG)",
        "description": "Retrieves medical knowledge to answer health-related questions using RAG",
        "url": "http://localhost:5000"
    },
    "gan": {
        "name": "Image Generator (CGAN)",
        "description": "Conditional GAN that generates images from user prompts",
        "url": "http://localhost:8000"
    },
    "glide": {
        "name": "Text-to-Image (GLIDE)",
        "description": "Diffusion-based model that generates photo-realistic images from text",
        "url": "http://localhost:8003"
    }
}

PAGES = {
    "Sentiment Transformer": "pages/sentiment_transformer.py",
    "Medical Chatbot (RAG)": "pages/rag_medical_chatbot.py",
    "Image Generator (CGAN)": "pages/cgan_image_generator.py",
    "Text-to-Image (GLIDE)": "pages/glide_diffusion.py"
}

APP_INFO = {
    "title": "Explore AI",
    "subtitle": "An immersive experience where you can engage with our deployed AI models"
}