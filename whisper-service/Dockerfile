# Этап 1: Сборка whisper.cpp с совместимыми флагами
FROM ubuntu:22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone https://github.com/ggerganov/whisper.cpp.git && \
    cd whisper.cpp && \
    mkdir build && cd build && \
    cmake .. \
        -DWHISPER_NO_AVX=ON \
        -DWHISPER_NO_AVX2=ON \
        -DWHISPER_NO_F16C=ON \
        -DWHISPER_NO_FMA=ON \
        -DCMAKE_C_FLAGS="-march=x86-64 -mtune=generic" && \
    make -j$(nproc)

RUN mkdir -p /whisper-dist/bin /whisper-dist/lib

RUN cp /workspace/whisper.cpp/build/bin/whisper-cli /whisper-dist/bin/
RUN cp /workspace/whisper.cpp/build/bin/main /whisper-dist/bin/
RUN cp /workspace/whisper.cpp/build/src/libwhisper.so.1.8.2 /whisper-dist/lib/
RUN cp /workspace/whisper.cpp/build/src/libwhisper.so.1 /whisper-dist/lib/
RUN cp /workspace/whisper.cpp/build/ggml/src/libggml.so /whisper-dist/lib/
RUN cp /workspace/whisper.cpp/build/ggml/src/libggml-base.so /whisper-dist/lib/
RUN cp /workspace/whisper.cpp/build/ggml/src/libggml-cpu.so /whisper-dist/lib/

RUN cd /whisper-dist/lib && \
    ln -s libwhisper.so.1.8.2 libwhisper.so

# Этап 2: Финальный образ
FROM ubuntu:22.04

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    python3 \
    wget \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Копируем whisper-cli и библиотеки из этапа сборки
COPY --from=builder /whisper-dist/bin/* /usr/local/bin/
COPY --from=builder /whisper-dist/lib/* /usr/local/lib/

# Обновляем кэш библиотек
RUN ldconfig

# Копируем ВСЕ файлы из текущей директории
COPY . .

###########################################
RUN wget "https://storage.yandexcloud.net/ai-models-stellarclaw/whisper/ggml_small.bin" -O ggml_small.bin

# Устанавливаем Python зависимости
RUN pip3 install --no-cache-dir -r requirements.txt

# Создаем директорию для временных файлов
RUN mkdir -p /tmp/whisper

# Проверяем что whisper-cli работает
RUN whisper-cli --help

# Открываем порт для FastAPI
EXPOSE 8004

# Запускаем приложение с unbuffered output
CMD ["python3", "-u", "app.py"]
