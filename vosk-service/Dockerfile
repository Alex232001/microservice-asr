
FROM python:3.12-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY ai_model/vosk-model-small-ru-0.22 /app/ai_model/vosk-model-small-ru-0.22
COPY app.py .

EXPOSE 8002

CMD ["python", "-u", "app.py"]
