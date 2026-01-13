import asyncio
import sounddevice as sd
from core.listener import Listener
from core.brain import Brain
from core.speaker import Speaker
from utils.audio_tools import resample_audio

async def main():
    # Selección de hardware (ID 9 recomendado para Pipewire)
    sd.default.device = [9, 9]
    
    # Inicialización de componentes
    listener = Listener()
    brain = Brain()
    speaker = Speaker()

    print("\n>>> SISTEMA INICIADO. Habla con la IA...")

    try:
        while True:
            audio = await listener.listen()
            text = await listener.transcribe(audio)
            
            if text:
                print(f"Tú: {text}")
                if "adiós" in text.lower(): break
                
                response = await brain.think(text)
                print(f"IA: {response}")
                await speaker.speak(response)
                
    except KeyboardInterrupt:
        print("\nCerrando agente...")

if __name__ == "__main__":
    asyncio.run(main())