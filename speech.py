# speech.py
import asyncio
import os
import uuid
import speech_recognition as sr
from typing import Optional
from fastapi import UploadFile

class SpeechProcessor:
    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()

    def speech_to_text(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        try:
            mic = sr.Microphone()
            with mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            return self.recognizer.recognize_google(audio)
        except Exception: return None

    async def audio_file_to_text(self, audio_file: UploadFile) -> Optional[str]:
        suffix = os.path.splitext(audio_file.filename or "")[-1].lower() or ".wav"
        temp_path = f"/tmp/{uuid.uuid4()}{suffix}"
        try:
            with open(temp_path, "wb") as f: f.write(await audio_file.read())
            with sr.AudioFile(temp_path) as source:
                audio = self.recognizer.record(source)
            return self.recognizer.recognize_google(audio)
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)
