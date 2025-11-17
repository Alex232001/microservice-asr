from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import numpy as np
import io
from typing import Dict
from os import path
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import wave
import json

#import os


app = FastAPI(title="Vosk Speech Recognition")

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

class VoskTranscriber:
    def __init__(self):
        self.model_loaded = True

        self.model = None
        self.load_model()
    def load_model(self) -> None:
        try:
            model_path = "ai_model/vosk-model-small-ru-0.22"
            if not path.exists(model_path):
                raise Exception(f"Model path {model_path} does not exist")
            
            self.model = Model(model_path)
        except Exception as e:
            print(f"Error loading VOSK model: {e}")
            raise

    def transcribe(self, wf) -> dict:
        start_time = time.time() 
        
        rec = KaldiRecognizer(self.model, wf.getframerate())
        rec.SetWords(True)

        results = []
        while True:
            data = wf.readframes(4096)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())

                if 'result' in result:
                    for word_info in result['result']:
                        results.append({
                            'word': word_info['word'],
                            'confidence': word_info['conf']
                        })

        final_result = json.loads(rec.FinalResult())
        if 'result' in final_result:
            for word_info in final_result['result']:
                results.append({
                    'word': word_info['word'],
                    'confidence': word_info['conf']
                })

        final_text = " ".join([word['word'] for word in results])
        avg_confidence = sum(word['confidence'] for word in results) / len(results) if results else 0
        avg_confidence2 = np.mean([word['confidence'] for word in results])

        end_time = time.time() 
        execution_time = end_time - start_time 
        return {
            "text": final_text,
            "processing_time": round(execution_time, 2),
            "confidence": round(avg_confidence, 2),
            "model": "vosk-ru"
        }

transcriber  = VoskTranscriber()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        
        audio_data = await file.read()
        
        if not is_audio_file(file.content_type, file.filename, audio_data):
            raise HTTPException(400, f"File must be an audio file. Got: {file.content_type or 'no content-type'}, filename: {file.filename}")
        
        if not file.filename.lower().endswith('.wav'):
            audio_data = convert_audio_to_wav(audio_data, file.filename)
       
       
        audio_buffer = io.BytesIO(audio_data)
        with wave.open(audio_buffer, "rb") as wf:
            start_time = time.time()
            result = transcriber.transcribe(wf)
            end_time = time.time()
            
            result["actual_processing_time"] = round(end_time - start_time, 2)
            result["file_size"] = len(audio_data)
            result["file_type"] = file.content_type
            
            return result
        
    except Exception as e:
        raise HTTPException(500, f"Transcription error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "vosk"}


@app.get("/")
async def root():
    return {"message": "Vosk Speech Recognition API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
