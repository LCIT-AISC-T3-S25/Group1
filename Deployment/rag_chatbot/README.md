# Memory-Optimized RAG Chatbot

A production-ready Retrieval-Augmented Generation (RAG) system specialized for medical queries, optimized for memory efficiency and Docker deployment.

## 🚀 Quick Start

### Prerequisites
- Docker installed on your system
- At least 4GB RAM available
- Optional: NVIDIA GPU with CUDA support
- Optional: Hugging Face account for gated models

### Docker Deployment

1. **Build the Docker image:**
```bash
docker build -t rag-chatbot .
```

2. **Run the container:**
```bash
docker run -p 5000:5000 rag-chatbot
```

3. **With Hugging Face token (for better models):**
```bash
docker run -p 5000:5000 -e HUGGINGFACE_HUB_TOKEN=your_token_here rag-chatbot
```

4. **Check health status:**
```bash
curl http://localhost:5000/health
```

## 📋 API Endpoints

### Health Check
```bash
GET /health
```
Returns system status and model loading information.

### Predict (Main Endpoint)
```bash
POST /predict
Content-Type: application/json

{
  "query": "What is diabetes?"
}
```

**Response Format:**
```json
{
  "response": "Generated medical response",
  "confidence": 0.85,
  "query": "What is diabetes?",
  "timestamp": "2024-01-15T10:30:00",
  "context_check": "passed",
  "model_used": "TinyLlama",
  "rewritten_query": "diabetes mellitus blood glucose insulin",
  "passages_found": 3
}
```

### Model Information
```bash
GET /info
```
Returns detailed model specifications and system information.

## 🏗️ Architecture

### Components
- **Embedder**: sentence-transformers/all-MiniLM-L6-v2
- **Query Rewriter**: google/gemma-2-2b-it (4-bit quantized)
- **Answer Generator**: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- **Vector Store**: FAISS index with 27,975 medical passages
- **Fallback Models**: DistilGPT2, T5-small

### Memory Optimizations
- 4-bit quantization using BitsAndBytes
- Lazy model loading
- GPU memory management
- Efficient passage retrieval

## 📁 Project Structure

```
RAG_SYSTEM_CHECKPOINT/
├── app.py                          # Flask API server
├── chatbot.py                      # Main RAG logic
├── load_system.py                  # System loader
├── Dockerfile                      # Docker configuration
├── requirements.txt                # Python dependencies
├── model-info.json                 # API specifications
├── system_config.json             # System configuration
└── rag_system_checkpoint/         # Pre-trained models & data
    ├── passages_list.pkl
    ├── passage_ids.pkl
    ├── bioasq_faiss_index.index
    └── system_config.json
```

## 🔧 Configuration

### Environment Variables
- `HUGGINGFACE_HUB_TOKEN`: HF token for gated models
- `FLASK_ENV`: Set to 'development' for debug mode
- `PORT`: Custom port (default: 5000)

### Model Configuration
Edit `system_config.json` to modify:
- Model parameters (temperature, max_tokens)
- Medical keyword filtering
- Retrieval settings

## 🧪 Testing

### Local Testing
```python
import requests

# Test health
response = requests.get('http://localhost:5000/health')
print(response.json())

# Test prediction
response = requests.post('http://localhost:5000/predict', 
                        json={'query': 'What is diabetes?'})
print(response.json())
```

### Docker Testing
```bash
# Test container health
docker exec <container_id> curl http://localhost:5000/health

# View logs
docker logs <container_id>
```

## 🐳 Docker Commands

### Build with custom tag
```bash
docker build -t my-rag-system:v1.0 .
```

### Run with custom configuration
```bash
docker run -p 5000:5000 \
  -e HUGGINGFACE_HUB_TOKEN=your_token \
  -e FLASK_ENV=production \
  -v $(pwd)/logs:/app/logs \
  rag-chatbot
```

### Save Docker image
```bash
docker save rag-chatbot > rag-chatbot.tar
```

### Load Docker image
```bash
docker load < rag-chatbot.tar
```

## 📊 Performance

- **Response Time**: 25 seconds average
- **Memory Usage**: ~3-4GB RAM
- **GPU Usage**: Optional, improves speed
- **Concurrent Users**: Limited by available memory
- **Model Size**: ~2GB compressed

## 🔒 Security & Limitations

### Security
- No user authentication (add as needed)
- Rate limiting not implemented
- Input validation for medical queries only

### Limitations
- **Domain**: Medical/health queries only
- **Language**: English only
- **Knowledge**: Based on pre-indexed passages
- **Real-time**: No live data updates

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Increase Docker memory allocation
   docker run -m 8g -p 5000:5000 rag-chatbot
   ```

2. **Model Loading Timeout**
   ```bash
   # Check logs for loading progress
   docker logs -f <container_id>
   ```

3. **CUDA Errors (GPU)**
   ```bash
   # Run CPU-only mode
   docker run -e CUDA_VISIBLE_DEVICES="" -p 5000:5000 rag-chatbot
   ```

### Health Check Failures
- Wait 2-3 minutes for complete model loading
- Check container logs for specific errors
- Verify sufficient memory allocation

## 📝 Development

### Adding New Models
1. Update `chatbot.py` model loading section
2. Modify `system_config.json` parameters  
3. Update `model-info.json` specifications
4. Rebuild Docker image

### Custom Data
1. Replace files in `rag_system_checkpoint/`
2. Update passage counts in config files
3. Rebuild FAISS index if needed

## 🤝 Assignment Requirements Compliance

✅ **Dockerized Model**: Complete Docker setup with exposed port 5000  
✅ **Health Endpoint**: `/health` returns system status  
✅ **Predict Endpoint**: `/predict` accepts JSON and returns confidence  
✅ **Documentation**: Comprehensive README.md  
✅ **Model Info**: Detailed model-info.json with specifications  
✅ **Best Practices**: Structured code, error handling, logging  

## Support

For issues or questions:
1. Check Docker logs: `docker logs <container_id>`
2. Verify system requirements (RAM, Docker version)
3. Test API endpoints manually with curl
4. Review this README for troubleshooting steps

---