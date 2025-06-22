# Use a minimal Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy your app code and model
COPY cnn_app.py ./cnn_app.py
COPY CNN_model_tuned2.keras ./CNN_model_tuned2.keras
COPY static/ ./static/

# Install only required dependencies
RUN pip install --no-cache-dir fastapi uvicorn tensorflow pillow python-multipart lime

# Expose the port your app runs on
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "cnn_app:app", "--host", "0.0.0.0", "--port", "8000"]
