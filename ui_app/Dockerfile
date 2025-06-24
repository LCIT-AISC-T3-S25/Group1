FROM python:3.11-slim

# Update system packages to fix vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean && \
	addgroup --system appgroup && adduser --system --ingroup appgroup appuser
# Set working directory
WORKDIR /app

# Install system-level dependencies for OpenCV, LIME, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


# Copy all application files
COPY . .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Switch to non-root user
USER appuser

# Expose Flask port
EXPOSE 5000

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

# Start the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
