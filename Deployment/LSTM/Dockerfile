FROM python:3.9-slim

WORKDIR /app

COPY app.py .
COPY LSTM_model.h5 .
COPY tokenizer.pkl .

RUN pip install flask tensorflow numpy pickle-mixin

EXPOSE 5000

CMD ["python", "app.py"]
