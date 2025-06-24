FROM python:3.9-slim

WORKDIR /app

COPY app.py .
COPY LSTM_model.h5 .
COPY tokenizer.pkl .

RUN pip install flask numpy pickle-mixin tensorflow

EXPOSE 5000

CMD ["python", "app.py"]
