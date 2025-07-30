FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8001 \
    MODELS_DIR=/app/app/models

# Install dependencies and create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
 && useradd --create-home --shell /bin/bash appuser \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app source and set ownership
COPY app /app/app
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8001

# Run the app using Gunicorn
CMD ["gunicorn", "-w", "2", "-k", "gthread", "--threads", "8", "-b", "0.0.0.0:8001", "app.main:app"]