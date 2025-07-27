# Memory-Optimized RAG Chatbot Docker Configuration
# Using Python 3.11 for better performance and latest features
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Create necessary directories
RUN mkdir -p data models logs uploads RAG_SYSTEM_CHECKPOINT

# Set proper permissions
RUN chmod -R 755 /app

# Health check with longer timeout for model loading
HEALTHCHECK --interval=60s --timeout=45s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port 5000 as required
EXPOSE 5000

# Start the application
CMD ["python", "app.py"]