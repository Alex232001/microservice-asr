from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort
import numpy as np
import librosa
import sentencepiece as spm
from typing import List, Dict, Optional
import time
from pydub import AudioSegment
import io
import os
import tempfile

app = FastAPI(title="ONNX Speech Recognition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все домены (для разработки)
    allow_credentials=True,
    allow_methods=["*"],  # Разрешаем все методы
    allow_headers=["*"],  # Разрешаем все заголовки
)

def is_audio_file(content_type: str, filename: str, file_content: bytes) -> bool:
    if content_type and content_type.startswith('audio/'):
        return True

    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
    file_extension = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
    if file_extension in audio_extensions:
        return True
    
    if not content_type and file_extension in audio_extensions:
        return True
    return False

def convert_audio_to_wav(audio_data: bytes, filename: str) -> bytes:
    try:
        file_extension = filename.split('.')[-1].lower()
        
        audio_buffer = io.BytesIO(audio_data)
        audio = AudioSegment.from_file(audio_buffer, format=file_extension)
        
        audio = audio.set_channels(1)  # моно
        audio = audio.set_frame_rate(16000)  # 16kHz
        audio = audio.set_sample_width(2)  # 16-bit

        output_buffer = io.BytesIO()
        audio.export(output_buffer, format="wav")
        
        return output_buffer.getvalue()  
    except Exception as e:
        raise Exception(f"Ошибка конвертации аудио: {str(e)}")

class AudioPreprocessor:
    """Препроцессор идентичный NeMo"""
    
    def __init__(self):
        self.sample_rate = 16000
        self.n_fft = 512
        self.win_length = 400  # 25ms
        self.hop_length = 160  # 10ms
        self.n_mels = 80
        self.window = 'hann'
        self.f_min = 0
        self.f_max = 8000
        self.dither = 1e-05
        self.preemph = 0.97
        self.log_zero_guard_value = 2**-24
        
    def compute_mel_spectrogram(self, audio):
        """Вычисление Mel-спектрограммы"""
        # Пре-эмфаза
        audio = np.append(audio[0], audio[1:] - self.preemph * audio[:-1])
        
        # STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True
        )
        
        # Амплитудный спектр
        magnitude = np.abs(stft)
        
        # Mel-фильтры
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            norm='slaney'
        )
        
        # Применение Mel-фильтров
        mel_spectrogram = np.dot(mel_basis, magnitude)
        
        # Логарифмирование
        log_mel = np.log(np.clip(mel_spectrogram, a_min=self.log_zero_guard_value, a_max=None))
        
        return log_mel
    
    def normalize_batch(self, features, seq_len):
        """Нормализация признаков"""
        mean = features.mean(axis=2, keepdims=True)
        std = features.std(axis=2, keepdims=True)
        normalized = (features - mean) / (std + 1e-5)
        return normalized, seq_len
    
    def __call__(self, audio_signal, audio_length):
        """Основной метод препроцессинга"""
        # Убедимся, что audio_signal - это numpy array
        if hasattr(audio_signal, 'numpy'):
            audio_signal = audio_signal.numpy()
        
        batch_size = audio_signal.shape[0]
        features_list = []
        features_lengths = []
        
        for i in range(batch_size):
            audio = audio_signal[i]
            length = audio_length[i]
            audio = audio[:length]
            
            mel_spec = self.compute_mel_spectrogram(audio)
            features_list.append(mel_spec)
            features_lengths.append(mel_spec.shape[1])
        
        # Собираем батч
        max_length = max(features_lengths)
        batch_features = np.zeros((batch_size, self.n_mels, max_length), dtype=np.float32)
        
        for i, feat in enumerate(features_list):
            batch_features[i, :, :feat.shape[1]] = feat
        
        features_lengths = np.array(features_lengths, dtype=np.int64)
        
        # Нормализация
        batch_features, features_lengths = self.normalize_batch(batch_features, features_lengths)
        
        return batch_features, features_lengths

def decode_with_sentencepiece(sp, tokens):
    """Декодирование токенов с помощью SentencePiece"""
    try:
        # Фильтруем токены - оставляем только валидные (0-127)
        valid_tokens = [t for t in tokens if t < sp.vocab_size()]
        
        if not valid_tokens:
            return ""
        
        # Пробуем разные способы декодирования
        try:
            # Способ 1: Прямое декодирование
            text = sp.decode(valid_tokens)
            return text
        except:
            # Способ 2: Декодирование через pieces
            pieces = []
            for token_id in valid_tokens:
                try:
                    piece = sp.id_to_piece(int(token_id))
                    pieces.append(piece)
                except:
                    continue
            
            if pieces:
                text = sp.decode_pieces(pieces)
                return text
            else:
                return ""
                
    except Exception as e:
        print(f"❌ Ошибка при декодировании: {e}")
        return ""

def greedy_batch_decode_with_probs(logprobs, lengths):
    """Жадное декодирование батча с возвратом вероятностей"""
    batch_size = logprobs.shape[0]
    predictions = []
    probabilities = []
    token_probs_list = []
    
    for i in range(batch_size):
        seq_len = lengths[i]
        if seq_len > logprobs.shape[1]:
            seq_len = logprobs.shape[1]
            
        seq_logprobs = logprobs[i, :seq_len]
        best_tokens = np.argmax(seq_logprobs, axis=1)
        best_probs = np.exp(np.max(seq_logprobs, axis=1))  # Преобразуем logprobs в вероятности
        
        # CTC декодирование - удаляем повторяющиеся токены и blank токены
        decoded_tokens = []
        decoded_probs = []
        token_probs = []  # Здесь сохраним все вероятности для каждого шага времени
        
        prev_token = -1
        
        for time_step, (token_idx, prob) in enumerate(zip(best_tokens, best_probs)):
            # Сохраняем информацию о всех токенах и их вероятностях на каждом шаге
            token_probs.append({
                'time_step': int(time_step),  # Преобразуем в int
                'token_id': int(token_idx),   # Преобразуем в int
                'token_prob': float(prob),    # Преобразуем в float
                'is_blank': bool(token_idx == 128),
                'is_repeated': bool(token_idx == prev_token)
            })
            
            if token_idx != prev_token:
                # Пропускаем blank токены (индекс 128)
                if token_idx != 128 and token_idx < 128:  # Только валидные токены
                    decoded_tokens.append(int(token_idx))  # Преобразуем в int
                    decoded_probs.append(float(prob))      # Преобразуем в float
                prev_token = token_idx
        
        predictions.append(decoded_tokens)
        probabilities.append(decoded_probs)
        token_probs_list.append(token_probs)
    
    return predictions, probabilities, token_probs_list

def convert_numpy_types(obj):
    """Рекурсивно преобразует NumPy типы в стандартные Python типы"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

class ONNXTranscriber:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.preprocessor = None
        self.load_model()

    def load_model(self) -> None:
        try:
            model_path = "model.onnx"
            tokenizer_path = "tokenizer.model"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX модель не найдена: {model_path}")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Токенизатор не найден: {tokenizer_path}")
            
            print("Загрузка ONNX модели и токенизатора...")
            self.model = ort.InferenceSession(model_path)
            self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
            self.preprocessor = AudioPreprocessor()
            
            print(f"✅ Модель загружена. Размер словаря: {self.tokenizer.vocab_size()}")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise

    def transcribe_audio(self, audio_data: bytes) -> dict:
        start_time = time.time()
        
        if not self.model or not self.tokenizer:
            print("Модель или токенизатор не загружены")
            return None
        
        try:
            # Сохраняем временный файл для обработки
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            # Загружаем и обрабатываем аудио
            audio, sr = librosa.load(temp_file_path, sr=16000)
            
            # Препроцессинг аудио без torch
            audio_array = np.expand_dims(audio, axis=0).astype(np.float32)
            audio_length = np.array([audio_array.shape[1]], dtype=np.int64)
            
            processed_audio, processed_audio_length = self.preprocessor(audio_array, audio_length)
            
            # Распознавание с ONNX моделью
            onnx_inputs = {
                'audio_signal': processed_audio.astype(np.float32),
                'length': processed_audio_length.astype(np.int64)
            }
            
            logprobs = self.model.run(None, onnx_inputs)[0]
            
            # Декодируем результат с вероятностями
            token_sequences, probability_sequences, all_token_probs = greedy_batch_decode_with_probs(
                logprobs, processed_audio_length
            )
            
            # Конвертируем токены в текст
            if token_sequences[0]:
                text_result = decode_with_sentencepiece(self.tokenizer, token_sequences[0])
                avg_confidence = np.mean(probability_sequences[0]) if probability_sequences[0] else 0.0
            else:
                text_result = ""
                avg_confidence = 0.0

            end_time = time.time()
            execution_time = end_time - start_time

            # Создаем результат и преобразуем все NumPy типы
            result = {
                'text': text_result,
                'processing_time': float(round(execution_time, 2)),  # Преобразуем в float
                'confidence': float(round(avg_confidence, 2)),       # Преобразуем в float
                'model': 'onnx-ru',
                'tokens_count': len(token_sequences[0]) if token_sequences[0] else 0
            }
            
            # Рекурсивно преобразуем все NumPy типы в стандартные Python типы
            return convert_numpy_types(result)
        
        except Exception as e:
            print(f"Ошибка транскрибации: {e}")
            return None
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def print_results(self, results: dict) -> None:
        if not results:
            print("Нет результатов для вывода")
            return
        print(f"Распознанный текст: {results['text']}")
        print(f"Уверенность: {results.get('confidence', 0):.2f}")
        print(f"Время обработки: {results.get('processing_time', 0)} сек")
        print(f"Модель: {results.get('model', 'неизвестно')}")
        print(f"Количество токенов: {results.get('tokens_count', 0)}")

# Инициализация транскрайбера
transcriber = ONNXTranscriber()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Проверяем файл
        if not file.filename:
            raise HTTPException(400, "No filename provided")
        
        audio_data = await file.read()
        
        # Проверка типа файла
        if not is_audio_file(file.content_type, file.filename, audio_data):
            raise HTTPException(400, f"File must be an audio file. Got: {file.content_type or 'no content-type'}, filename: {file.filename}")
        
        # Конвертируем в WAV если нужно
        if not file.filename.lower().endswith('.wav'):
            audio_data = convert_audio_to_wav(audio_data, file.filename)
       
        # Транскрибация
        start_time = time.time()
        result = transcriber.transcribe_audio(audio_data)
        end_time = time.time()
        
        if result is None:
            raise HTTPException(500, "Transcription failed")
            
        # Добавляем дополнительную информацию (преобразуем в стандартные типы)
        result["actual_processing_time"] = float(round(end_time - start_time, 2))
        result["file_size"] = int(len(audio_data))  # Преобразуем в int
        result["file_type"] = file.content_type
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Transcription error: {str(e)}")

@app.get("/health")
async def health_check():
    status = "healthy" if transcriber.model and transcriber.tokenizer else "unhealthy"
    return {"status": status, "model": "onnx"}

@app.get("/")
async def root():
    return {"message": "ONNX Speech Recognition API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)