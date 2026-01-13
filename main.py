import asyncio
import torch
import sounddevice as sd
import numpy as np
from kokoro import KPipeline
from faster_whisper import WhisperModel
from ollama import AsyncClient

class local_voice_agent:
    def __init__(self):
        # 1. Configuración de Modelos (Aprovechando los 16GB de VRAM)
        print("--- Inicializando Motores en RTX 5060 ---")
        
        # Escucha: Usamos modelo 'small' para equilibrio entre precisión y velocidad
        self.stt_model = WhisperModel("medium", device="cuda", compute_type="float16")
        
        # Pensamiento: Gemma 3 vía Ollama
        self.ollama = AsyncClient()
        self.llm_model = "gemma3:12b"
        
        # Habla: Kokoro (Asegúrate de tener instalada la versión >= 0.9.4)
        self.tts_pipeline = KPipeline(lang_code='es')
        
        self.sample_rate = 16000 # Estándar para Whisper
        self.is_active = True

    def select_audio_hardware(self):
        """Cumple con tu requerimiento de entrada/salida de audio."""
        print("\n--- DISPOSITIVOS DETECTADOS ---")
        devices = sd.query_devices()
        print(devices)
        in_id = int(input("\nID del Micrófono (Entrada): "))
        out_id = int(input("ID de Altavoces (Salida): "))
        return in_id, out_id

    async def record_and_transcribe(self):
        duration = 5 
        print("\n[Escuchando...]")
    
        try:
        # Grabamos a 48kHz. Forzamos channels=1 para evitar ambigüedad
            audio_data = sd.rec(int(duration * self.sample_rate), 
                            samplerate=self.sample_rate, 
                            channels=1, dtype='float32')
            sd.wait()
        
        # Eliminamos dimensiones extra (Whisper espera un array 1D)
            audio_flat = audio_data.flatten()
        
            # Transcribimos. Faster-Whisper remuestrea de 48k a 16k automáticamente
            segments, _ = self.stt_model.transcribe(audio_flat, language="es")
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            print(f"Error capturando audio: {e}")
            return ""

    async def generate_ai_response(self, user_text):
        """Consulta a Gemma 3 con un System Prompt optimizado."""
        messages = [
            {'role': 'system', 'content': 'Eres una IA local concisa. Responde en español natural. Máximo 2 frases.'},
            {'role': 'user', 'content': user_text}
        ]
        response = await self.ollama.chat(model=self.llm_model, messages=messages)
        return response['message']['content']

    async def speak(self, text):
        """Genera audio con Kokoro y lo re-muestrea para el hardware USB."""
        print(f"IA: {text}")
        try:
            # Generamos el audio (Kokoro entrega 24000 Hz)
            generator = self.tts_pipeline(text, voice='em_alex', speed=1.1)
        
            for _, _, audio in generator:
                # --- SOLUCIÓN AL ERROR 9997 (SALIDA) ---
                # Re-muestreo de 24000 a 44100 usando interpolación lineal de numpy
                # Esto adapta el audio de la IA a la velocidad de tu tarjeta USB
                target_sr = self.sample_rate  # Usamos los 44100 o 48000 que configuramos antes
                source_sr = 24000
            
                num_samples = int(len(audio) * target_sr / source_sr)
                audio_resampled = np.interp(
                    np.linspace(0, len(audio), num_samples, endpoint=False),
                    np.arange(len(audio)),
                    audio
                ).astype(np.float32)

                # Reproducimos a la tasa que el hardware SÍ acepta
                sd.play(audio_resampled, target_sr)
                sd.wait()
            
        except Exception as e:
            print(f"Error en la reproducción de voz: {e}")

    async def start(self):
        in_id, out_id = self.select_audio_hardware()
        sd.default.device = [in_id, out_id]
        
        print("\n>>> AGENTE INICIADO. Di 'Adiós' para salir.")
        
        while self.is_active:
            user_text = await self.record_and_transcribe()
            
            if user_text:
                print(f"Tú: {user_text}")
                
                if "adiós" in user_text.lower():
                    await self.speak("Hasta pronto. Cerrando sistemas.")
                    self.is_active = False
                    break
                
                # Proceso de pensamiento y habla
                ai_response = await self.generate_ai_response(user_text)
                await self.speak(ai_response)
            else:
                print("... no detecté voz.")

if __name__ == "__main__":
    agent = local_voice_agent()
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nAgente detenido manualmente.")