from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import nemo.collections.asr as nemo_asr
import torch
import numpy as np
from typing import List, Tuple, Optional,Dict
import time

from pydub import AudioSegment
import io
from os import path
import os
import tempfile

app = FastAPI(title="Nemo Speech Recognition")

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


class NemoTranscriber:

    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        try:
            model_path = "ai_model/stt_ru_conformer_ctc_large"
            if os.path.exists(model_path):
                print(f"Загружаем модель из локального файла: {model_path}")
                self.model = nemo_asr.models.EncDecCTCModelBPE.restore_from(model_path)
            else:
               self.model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                model_name="stt_ru_conformer_ctc_large")
        except Exception as e:
            print(f"Модель не загружена {e}")
            raise
    
    def transcribe_audio(self, audio_data: bytes) -> dict:
        start_time = time.time() 
        if not self.model:
            print("Модель не загружена")
            return None
        
        try:

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            predictions = self.model.transcribe([temp_file_path], return_hypotheses=True)

            os.unlink(temp_file_path)

            text = predictions[0].text
            words = text.split()
            
            # Извлечение токенов и вероятностей
            tokens, probabilities = self.tokens_and_probs(predictions[0])
            
            # Расчет уверенности для каждого слова
            word_confidence = self.word_confidence(words, probabilities)

            avg_confidence2 = np.mean([wc['confidence'] for wc in word_confidence])

            end_time = time.time() 
            execution_time = end_time - start_time 
            return {
                'text': text,
                "processing_time": round(execution_time, 2),
                "confidence": round(avg_confidence2, 2),
                "model": "nemo-ru"
            }
        
        except Exception as e:
            print(f"Ошибка транскрибации: {e}")
            return None
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
     
    def tokens_and_probs(self, hypothesis) -> Tuple[List[int], List[float]]:

        if not hasattr(hypothesis, 'alignments'):
            return [], []
        
        tokens = torch.argmax(hypothesis.alignments, dim=-1)
        probs = torch.softmax(hypothesis.alignments, dim=-1)
        
        decoded_tokens = []
        decoded_probs = []
        prev_token = -1
        
        for i, token in enumerate(tokens):
            token_item = token.item()
            # Пропускаем blank-токены
            if token_item != prev_token and token_item != 0:
                decoded_tokens.append(token_item)
                decoded_probs.append(probs[i, token].item())
            prev_token = token_item
        
        return decoded_tokens, decoded_probs
    
    def word_confidence(self, words: List[str], probabilities: List[float]) -> List[dict]:

        if not probabilities:
            return [{'word': word, 'confidence': 0.5} for word in words]
        
        avg_tokens_per_word = len(probabilities) / len(words)
        word_confidence = []
        
        for i, word in enumerate(words):
            start_idx = int(i * avg_tokens_per_word)
            end_idx = min(int((i + 1) * avg_tokens_per_word), len(probabilities))
            
            if start_idx < end_idx:
                word_probs = probabilities[start_idx:end_idx]
                confidence = np.mean(word_probs)
            else:
                confidence = 0.5
             
            word_confidence.append({
                'word': word,
                'confidence': confidence,

            })
        
        return word_confidence
    
    def print_results(self, results: dict) -> None:
        if not results:
            print("Нет результатов для вывода")
            return
        print(f"Распознанный текст: {results['text']}")
        print(f"Уверенность: {results.get('confidence', 0):.2f}")
        print(f"Время обработки: {results.get('processing_time', 0)} сек")
        print(f"Модель: {results.get('model', 'неизвестно')}")


transcriber = NemoTranscriber()



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
            
        result["actual_processing_time"] = round(end_time - start_time, 2)
        result["file_size"] = len(audio_data)
        result["file_type"] = file.content_type
            
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Transcription error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "nemo"}

# Добавь root endpoint для тестирования
@app.get("/")
async def root():
    return {"message": "Nemo Speech Recognition API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)


