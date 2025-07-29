import streamlit as st
import requests

st.title("Biomedical RAG Chatbot")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear button
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Input box
user_input = st.chat_input("Ask a biomedical question...")

# API endpoint from model-info.json
API_URL = "http://localhost:5000/predict"

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    try:
        response = requests.post(API_URL, json={"query": user_input})
        res = response.json()

        bot_response = res.get("response", "No response.")
        confidence = res.get("confidence", None)
        rewritten = res.get("rewritten_query", None)
        context_check = res.get("context_check", "unknown")

        # Append bot reply to chat
        st.session_state.chat_history.append({
            "role": "bot",
            "content": bot_response,
            "confidence": confidence,
            "rewritten_query": rewritten,
            "context_check": context_check
        })

    except Exception as e:
        st.session_state.chat_history.append({
            "role": "bot",
            "content": f"❌ Error contacting backend: {e}"
        })

# Display chat history
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        with st.chat_message("🧑‍⚕️ You"):
            st.markdown(entry["content"])
    else:
        with st.chat_message("🤖 Bot"):
            st.markdown(f"**Answer:** {entry['content']}")
            if "confidence" in entry and entry["confidence"] is not None:
                st.markdown(f"**Confidence:** {entry['confidence']:.2f}")
            if entry.get("rewritten_query"):
                st.markdown(f"**Rewritten Query:** {entry['rewritten_query']}")
            if entry.get("context_check") == "failed":
                st.warning("⚠️ This question might be out of context.")
