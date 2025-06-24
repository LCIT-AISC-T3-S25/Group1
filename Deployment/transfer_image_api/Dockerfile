# Use minimal Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only necessary files from local context (relative paths)
COPY vgg_app.py ./vgg_app.py
COPY vgg_finetuned_round2.pth ./vgg_finetuned_round2.pth
COPY static/ ./static/

# Install dependencies (CPU only)
RUN pip install --no-cache-dir fastapi pillow python-multipart torch torchvision uvicorn

# Expose app port
EXPOSE 8000

# Start server
CMD ["uvicorn", "vgg_app:app", "--host", "0.0.0.0", "--port", "8000"]
