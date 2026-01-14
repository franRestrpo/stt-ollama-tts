# --- 4. ORQUESTADOR (Main Pipeline) ---
import asyncio
import sounddevice as sd

from core.listener import Listener
from core.brain import Brain
from core.speaker import Speaker
class VoiceAgent:
    def __init__(self):
        # Asignamos ID 12 como default para Pipewire/Linux
        sd.default.device = [12, 12]
        self.listener = Listener()
        self.brain = Brain()
        self.speaker = Speaker()

    async def run(self):
        print("\n>>> AGENTE ACTIVO (Gemma 3 + Whisper Medium)")
        try:
            while True:
                audio = await self.listener.get_audio_input()
                text = await self.listener.transcribe(audio)
                
                if text:
                    print(f"Tú: {text}")
                    if "adiós" in text.lower():
                        await self.speaker.play_audio("Hasta luego, ha sido un placer.")
                        break
                    
                    response = await self.brain.generate_response(text)
                    await self.speaker.play_audio(response)
        except Exception as e:
            print(f"[!] Error en el pipeline: {e}")

if __name__ == "__main__":
    agent = VoiceAgent()
    asyncio.run(agent.run())