import asyncio
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from ollama import AsyncClient
# ... otras importaciones (Kokoro, etc.)

class AdvancedVoiceAgent:
    def __init__(self):
        # Usamos MEDIUM como pediste, en la RTX 5060
        self.stt = WhisperModel("medium", device="cuda", compute_type="float16")
        self.sample_rate = 16000 # Para VAD interno es mejor 16k, Pipewire hará el resto
        self.silence_threshold = 0.5 # Segundos de silencio para cerrar frase
        self.padding_duration = 0.3 # MARGEN DE SEGURIDAD para no cortar palabras

    async def get_voice_input(self):
        """Captura audio dinámicamente hasta que detecta un silencio final."""
        print("\n[Escuchando... hable ahora]")
        audio_buffer = []
        silence_counter = 0
        recording = False
        
        # Stream de audio con callback
        def callback(indata, frames, time, status):
            nonlocal recording, silence_counter
            volume = np.linalg.norm(indata) * 10
            
            if volume > 0.5: # Umbral de voz detectada
                recording = True
                silence_counter = 0
                audio_buffer.append(indata.copy())
            elif recording:
                silence_counter += frames / self.sample_rate
                audio_buffer.append(indata.copy())

        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback):
            while not recording or silence_counter < self.silence_threshold:
                await asyncio.sleep(0.1)
                if len(audio_buffer) > 500: # Timeout de seguridad (aprox 30 seg)
                    break

        print("[Procesando...]")
        # Unimos el audio y añadimos el padding para evitar cortes
        return np.concatenate(audio_buffer).flatten()

    async def transcribe_with_integrity(self, audio_data):
        """Transcribe asegurando que la frase esté completa."""
        # El vad_filter elimina ruidos antes/después, pero conservamos el contenido
        segments, info = self.stt.transcribe(
            audio_data, 
            beam_size=5, 
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500) 
        )
        
        text = " ".join([s.text for s in segments]).strip()
        
        # Filtro de alucinaciones mejorado para modelo Medium
        if info.language_probability < 0.5 or len(text) < 2:
            return ""
            
        return text

    async def start(self):
        # Usamos ID 12 (default) para evitar conflictos de hardware
        sd.default.device = [12, 12]
        
        while True:
            audio = await self.get_voice_input()
            user_text = await self.transcribe_with_integrity(audio)
            
            if user_text:
                print(f"Tú: {user_text}")
                # Aquí sigue el flujo hacia Ollama y Kokoro...