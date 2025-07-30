FROM python:3.11-slim

# Create a non-root user
RUN adduser --disabled-password --gecos '' appuser

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install only what's necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Copy dependencies first for caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# Create necessary folders and adjust permissions
RUN mkdir -p data models logs uploads RAG_SYSTEM_CHECKPOINT \
 && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Healthcheck for Flask API
HEALTHCHECK --interval=60s --timeout=45s --start-period=120s --retries=3 \
 CMD curl -f http://localhost:5000/health || exit 1

EXPOSE 5000

CMD ["python", "app.py"]
