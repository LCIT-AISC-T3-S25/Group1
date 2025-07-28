
# Centralized Streamlit UI for RAG, GAN, and GLIDE Models

This repository contains the **frontend UI** built using **Streamlit**, designed to interact with the following deployed model backends:

-  RAG Chatbot (FastAPI or Flask)
-  Conditional GAN Image Generator (FastAPI)
- 🖼 GLIDE Text-to-Image Diffusion (FastAPI)

---

##  How to Run UI Locally

1. **Install dependencies**
```bash
pip install -r requirements.txt
````

2. **Run Streamlit**

```bash
streamlit run ragchat_ui.py       # For RAG Chatbot
streamlit run gan_ui.py           # For CGAN
streamlit run glide_ui.py         # For GLIDE
```

3. **Access in browser**
   Go to: [http://localhost:8501](http://localhost:8501)

---

## 🔗 Backend API Requirements

Please ensure the following Docker containers are running:

| Model       | Endpoint                                   | Port |
| ----------- | ------------------------------------------ | ---- |
| RAG Chatbot | `http://localhost:5000/predict`            | 5000 |
| CGAN        | `http://localhost:8002/generate?label=...` | 8002 |
| GLIDE       | `http://localhost:8004/generate`           | 8004 |

---

## 🖥 Functionality

###  RAG Chatbot UI (`ragchat_ui.py`)

* Input: Biomedical question
* Output: Generated answer with confidence, rewritten query, and metadata

###  CGAN UI (`gan_ui.py`)

* Input: Dropdown label (Food, Drink, Inside, Outside)
* Output: PNG image generated using GAN
* *Note:* Uses `use_container_width=True` (Streamlit >=1.34)

### GLIDE UI (`glide_ui.py`)

* Input: Text prompt
* Output: AI-generated image using text-to-image diffusion

---

## 📌 Submission Notes

* This UI was developed to integrate with the provided model-info.json specifications.
* All Docker backends must be tested locally before running the UI.


