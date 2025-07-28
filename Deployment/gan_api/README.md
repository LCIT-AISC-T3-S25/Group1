# Conditional GAN Image Generator (CGAN)

A production-ready Conditional GAN (CGAN) model to generate images for 4 different classes (labels 0-3). The project includes a FastAPI server and Docker support for easy deployment.

---

## 🚀 Quick Start

### Prerequisites
- Docker installed on your system
- Python 3.10 (for local testing)
- Basic knowledge of FastAPI endpoints

---

### 🐳 Docker Deployment

1. **Build the Docker image:**
```bash
docker build -t gan_image_gen .
```

2. **Run the container:**
```bash
docker run -p 8002:8002 --env-file .env gan_image_gen
```

3. **Check health status:**
```bash
curl http://localhost:8002/
```

---

## 📋 API Endpoints

### Health Check
```bash
GET /
```
**Response:**
```json
{ "message": "CGAN API running" }
```

---

### Generate Image
```bash
GET /generate?label=<class_label>
```

**Query Parameter:**
- `label` → integer (0, 1, 2, 3) – Class label for image generation

**Example Request:**
```bash
curl -X GET "http://localhost:8002/generate?label=2" --output generated.png
```

**Response:** A PNG image generated for the specified label.

---

## 🏗️ Architecture

- **Model:** Conditional GAN (Generator + Discriminator)
- **Framework:** PyTorch
- **Input:** Random noise vector + label embedding
- **Latent Dimension:** 100
- **Classes:** 4 (labels 0–3)
- **Image Size:** 64×64
- **Channels:** 3 (RGB)

---

## 📁 Project Structure

```
gan_image_gen/
├── app/                          
│   ├── app.py                 # FastAPI application
│   ├── config.py              # Environment config
│   ├── utils.py               # Image generation utility
│   ├── model_architecture.py  # CGAN architecture
├── model/
│   └── cgan_complete.pth      # Trained model weights
├── requirements.txt
├── Dockerfile
├── .env
└── README.md
```

---

## 🔧 Configuration

### Environment Variables (`.env`)
```
MODEL_PATH=./model/cgan_complete.pth
LATENT_DIM=100
IMAGE_SIZE=64
HOST=0.0.0.0
PORT=8002
DEBUG=True
```

---

## 🧪 Local Testing (Without Docker)

```python
import requests

# Health Check
print(requests.get("http://localhost:8002/").json())

# Generate Image
response = requests.get("http://localhost:8002/generate?label=1")
with open("generated.png", "wb") as f:
    f.write(response.content)
```

---

## 🐳 Docker Commands

### Build with Custom Tag
```bash
docker build -t my-cgan-model:v1.0 .
```

### Run with Custom Config
```bash
docker run -p 8002:8002 --env-file .env my-cgan-model:v1.0
```

### Save Docker Image
```bash
docker save my-cgan-model:v1.0 > cgan_model.tar
```

### Load Docker Image
```bash
docker load < cgan_model.tar
```

---

## 📊 Performance

- **Average Generation Time:** 2-3 seconds
- **Memory Usage:** ~500 MB
- **Supports:** Labels 0-3
- **Output:** 64×64 RGB images

---

## 🔒 Limitations

- Only generates images for **labels 0-3** (as per training)
- Outputs differ each time due to random noise input
- Image quality depends on training dataset

---

## ✅ Features

✅ **Dockerized FastAPI app**  
✅ **Single endpoint `/generate` for image generation**  
✅ **Supports conditional generation based on label input**  
✅ **Ready for deployment in any environment**

---
