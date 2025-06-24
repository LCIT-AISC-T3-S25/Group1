FROM python:3.9-slim-bookworm

# Update system packages to fix vulnerabilities
RUN apt-get update && apt-get upgrade -y && apt-get clean && \
	addgroup --system appgroup && adduser --system --ingroup appgroup appuser
# Set working directory
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt && \
	chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

EXPOSE 5000
CMD ["python", "app.py"]


