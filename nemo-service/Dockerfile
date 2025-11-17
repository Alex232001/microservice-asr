FROM python:3.12-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Копируем установленные пакеты
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем код приложения
COPY app.py .

#################################################################
RUN wget "https://storage.yandexcloud.net/ai-models-stellarclaw/nemo/model.onnx" -O model.onnx

RUN wget "https://storage.yandexcloud.net/ai-models-stellarclaw/nemo/tokenizer.model" -O tokenizer.model
# COPY src/ ./src/

EXPOSE 8003

CMD ["python", "-u", "app.py"]