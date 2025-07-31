import streamlit as st
import requests
from datetime import datetime
from config import ENDPOINTS

st.set_page_config(page_title="RAG Medical Chatbot", page_icon="🧠")
st.title("🧠 RAG Medical Chatbot")

# Backend API from config
API_URL = ENDPOINTS["rag"]["url"]

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear history button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Chat input
user_input = st.chat_input("Ask a biomedical question...")

if user_input:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save user input
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })

    try:
        with st.spinner("🤖 Thinking..."):
            response = requests.post(API_URL, json={"query": user_input}, timeout=100)
            data = response.json()
            bot_response = data.get("response", "No response.")
    except Exception as e:
        bot_response = f"❌ Error contacting backend: {e}"

    # Save bot response
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_history.append({
        "role": "bot",
        "content": bot_response,
        "timestamp": timestamp
    })

# Display history
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        with st.chat_message("🧑‍⚕️", avatar="🧑‍⚕️"):
            st.markdown(f"**You**  \n_{entry['timestamp']}_")
            st.markdown(entry["content"])
    else:
        with st.chat_message("🤖", avatar="🤖"):
            st.markdown(f"**Bot**  \n_{entry['timestamp']}_")
            st.markdown(entry["content"])
