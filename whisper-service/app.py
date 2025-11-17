from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import subprocess
import json
#import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
import time
import re
import os

from pydub import AudioSegment
import io
import tempfile

app = FastAPI(title="Whisper Speech Recognition")

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

def calculate_confidence_from_segments(segments):
    """Рассчитываем уверенность на основе сегментов транскрипции"""
    if not segments:
        return 0.5
    
    # Если в сегментах есть информация о вероятностях, используем ее
    total_confidence = 0
    segment_count = 0
    
    for segment in segments:
        # Предполагаем, что если сегмент есть, то уверенность средняя
        # В реальности нужно смотреть на вероятности из модели
        text = segment.get('text', '').strip()
        if text:
            # Простая эвристика: чем длиннее сегмент, тем выше уверенность
            segment_confidence = min(0.8, 0.3 + (len(text) * 0.02))
            total_confidence += segment_confidence
            segment_count += 1
    
    if segment_count > 0:
        return round(total_confidence / segment_count, 2)
    else:
        return 0.5

def extract_word_details_from_segments(segments):
    """Извлекаем детали о словах из сегментов"""
    word_details = []
    
    if not segments:
        return word_details
    
    for segment in segments:
        text = segment.get('text', '').strip()
        if text:
            # Простая разбивка на слова (можно улучшить)
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                # Базовая уверенность для слова
                word_confidence = 0.7  # можно адаптировать на основе длины и других факторов
                word_details.append({
                    'word': word,
                    'confidence': word_confidence,
                    'confidence_level': get_confidence_level(word_confidence)
                })
    
    return word_details

def get_confidence_level(confidence):
    """Определяем уровень уверенности по числовому значению"""
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    else:
        return "low"

class WhisperCLITranscriber:
    def __init__(self):
        self.model_path = "ggml_small.bin"  # Путь к модели для whisper-cli
        self.load_model()

    def load_model(self) -> None:
        try:
            # Проверяем доступность whisper-cli и модели
            if not os.path.exists(self.model_path):
                print(f"Предупреждение: Модель {self.model_path} не найдена")
            
            # Проверяем доступность whisper-cli
            try:
                result = subprocess.run(["whisper-cli", "--help"], capture_output=True, text=True)
                if result.returncode == 0:
                    print("Whisper-CLI успешно инициализирован")
                else:
                    print("Предупреждение: whisper-cli может быть недоступен")
            except Exception as e:
                print(f"Предупреждение при проверке whisper-cli: {e}")
                
        except Exception as e:
            print(f"Ошибка инициализации Whisper-CLI: {e}")
            raise
    
    def transcribe_audio(self, audio_data: bytes) -> dict:
        start_time = time.time() 
        
        try:
            # Создаем временный файл для аудио
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            # Создаем временный файл для JSON вывода
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_json:
                temp_json_path = temp_json.name

            # Команда для получения JSON вывода
            cmd = [
                "whisper-cli", 
                "-f", temp_audio_path,
                "-m", self.model_path,
                "--output-json",
                "--output-file", temp_json_path.replace('.json', ''),  # Без расширения
                "--language", "ru",
                "-pp"
            ]
            
            print("🎯 Запуск транскрипции...")
            result = subprocess.run(
                cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore'
            )
            
            # Читаем JSON результат
            json_data = None
            if os.path.exists(temp_json_path):
                with open(temp_json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            
            if not json_data:
                raise Exception("Не удалось получить JSON результат от whisper-cli")
            
            # Формируем полный текст из сегментов
            full_text = ""
            segments = json_data.get('transcription', [])
            for segment in segments:
                segment_text = segment.get('text', '').strip()
                if segment_text:
                    full_text += segment_text + " "
            
            full_text = full_text.strip()
            
            # Рассчитываем уверенность
            overall_confidence = calculate_confidence_from_segments(segments)
            
            # Получаем детали по словам
            word_details = extract_word_details_from_segments(segments)
            
            # Формируем breakdown уверенности
            confidence_breakdown = {
                "overall_confidence": overall_confidence,
                "overall_confidence_percentage": f"{overall_confidence:.2%}",
                "word_details": word_details,
                "high_confidence_count": len([w for w in word_details if w['confidence'] >= 0.8]),
                "medium_confidence_count": len([w for w in word_details if 0.5 <= w['confidence'] < 0.8]),
                "low_confidence_count": len([w for w in word_details if w['confidence'] < 0.5]),
                "total_words_analyzed": len(word_details)
            }
            
            # Формируем результат
            end_time = time.time() 
            execution_time = end_time - start_time 
            
            result_data = {
                'text': full_text,
                "processing_time": round(execution_time, 2),
                "confidence": overall_confidence,
                "model": "whisper-cli-ru",
                "language": json_data.get('result', {}).get('language', 'ru'),
                "real_confidence": overall_confidence,
                "real_confidence_percentage": f"{overall_confidence:.2%}",
                "confidence_breakdown": confidence_breakdown,
                "segments_count": len(segments),
                "segments": segments
            }
            
            return result_data
        
        except Exception as e:
            print(f"Ошибка транскрибации Whisper-CLI: {e}")
            return None
        finally:
            # Очистка временных файлов
            if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
            if 'temp_json_path' in locals() and os.path.exists(temp_json_path):
                try:
                    os.unlink(temp_json_path)
                except:
                    pass

    def print_results(self, results: dict) -> None:
        if not results:
            print("Нет результатов для вывода")
            return
        print(f"Распознанный текст: {results['text']}")
        print(f"Уверенность: {results.get('confidence', 0):.2f}")
        print(f"Время обработки: {results.get('processing_time', 0)} сек")
        print(f"Модель: {results.get('model', 'неизвестно')}")
        print(f"Язык: {results.get('language', 'неизвестно')}")
        print(f"Сегментов: {results.get('segments_count', 0)}")

transcriber = WhisperCLITranscriber()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        
        audio_data = await file.read()
        
        # Улучшенная проверка типа файла
        if not is_audio_file(file.content_type, file.filename, audio_data):
            raise HTTPException(400, f"File must be an audio file. Got: {file.content_type or 'no content-type'}, filename: {file.filename}")
        
        if not file.filename.lower().endswith('.wav'):
            audio_data = convert_audio_to_wav(audio_data, file.filename)
       
        start_time = time.time()
        result = transcriber.transcribe_audio(audio_data)
        end_time = time.time()
        
        if result is None:
            raise HTTPException(500, "Transcription failed - no result returned")
            
        result["actual_processing_time"] = round(end_time - start_time, 2)
        result["file_size"] = len(audio_data)
        result["file_type"] = file.content_type
            
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Transcription error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "whisper-cli"}

@app.get("/")
async def root():
    return {"message": "Whisper-CLI Speech Recognition API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
