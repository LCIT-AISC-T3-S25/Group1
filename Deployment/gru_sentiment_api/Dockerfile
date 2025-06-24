FROM python:3.9-slim-bookworm

# Update system packages to fix vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean && \
	addgroup --system appgroup && adduser --system --ingroup appgroup appuser
# Set working directory
WORKDIR /app
# Copy only the required files
COPY app.py .
COPY utils.py .
COPY GRU_model.keras .
COPY tokenizer.pkl .
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to avoid caching and reduce image size
# Use chown to set ownership of the app directory to the non-root user
RUN pip install --no-cache-dir -r requirements.txt && \
	chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

EXPOSE 5000
CMD ["python", "app.py"]


