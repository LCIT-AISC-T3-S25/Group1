FROM python:3.10

# Create a non-root user and set up working dir
RUN useradd --create-home appuser && mkdir -p /app && chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy and install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files directly with ownership
COPY --chown=appuser:appuser ./src ./package.json ./requirements.txt /app/

# Switch to non-root user
USER appuser

EXPOSE 8002

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8002"]