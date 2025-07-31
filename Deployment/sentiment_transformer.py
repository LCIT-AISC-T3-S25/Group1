import streamlit as st
import requests

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")
st.title("💬 Sentiment Transformer")
st.caption("Enter a sentence and get its sentiment prediction from a transformer model.")

API_BASE = "http://sentiment-transformer:8001"

  # Updated port and base URL

# Health Check
try:
    health = requests.get(f"{API_BASE}/health")
    if health.status_code != 200:
        st.warning("⚠️ Sentiment backend is running but not healthy.")
except Exception:
    st.error("❌ Sentiment transformer backend not reachable.")
    st.stop()

# Input
user_input = st.text_area("Enter your sentence:", height=150)

# Predict Sentiment
if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a sentence to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                response = requests.post(
                    f"{API_BASE}/predict",
                    json={"text": user_input},
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()["results"][0]
                    sentiment = data.get("pred_label", "Unknown")
                    st.success(f"**Sentiment:** {sentiment}")
                    st.markdown(f"**Confidence:** {max(data.get('probs', [])):.2f}")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Request failed: {str(e)}")

# LIME Explanation
if st.button("Explain with LIME"):
    if not user_input.strip():
        st.warning("Please enter a sentence first.")
    else:
        with st.spinner("Explaining using LIME..."):
            try:
                response = requests.post(
                    f"{API_BASE}/explain",
                    json={"text": user_input},
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    st.subheader(f"Explanation for: _{data['text']}_")
                    st.markdown(f"**Predicted Sentiment:** {data['target_class_name']}")
                    st.write("**Important Words:**")
                    for word in data["weights"]:
                        st.markdown(f"`{word['term']}` → weight: `{word['weight']:.4f}`")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Explanation failed: {str(e)}")
