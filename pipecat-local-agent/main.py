import asyncio
import logging
import signal
import sys
import pyaudio

from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.frames.frames import StartFrame
from pipecat.audio.vad.silero import SileroVADAnalyzer

from pipecat.processors.aggregators.llm_response import (
    LLMUserContextAggregator, 
    LLMAssistantContextAggregator
)
from pipecat.processors.aggregators.sentence import SentenceAggregator

from services.whisper_stt import LocalWhisperService
from services.gemma_llm import LocalGemmaService
from services.kokoro_tts import LocalKokoroService


def select_device(p, is_input):
    """Permite al usuario seleccionar un dispositivo de audio desde la terminal."""
    type_str = "entrada (Micrófono)" if is_input else "salida (Altavoces)"
    print(f"\nScanning {type_str} devices...")
    
    count = p.get_device_count()
    devices = []
    
    print(f"\n--- DISPOSITIVOS DE {type_str.upper()} DISPONIBLES ---")
    idx_counter = 1
    for i in range(count):
        dev = p.get_device_info_by_index(i)
        channels = dev['maxInputChannels'] if is_input else dev['maxOutputChannels']
        if channels > 0:
            devices.append((i, dev['name']))
            print(f"{idx_counter}. {dev['name']}")
            idx_counter += 1
            
    if not devices:
        print(f"Error: No se encontraron dispositivos de {type_str}.")
        sys.exit(1)

    while True:
        try:
            choice = input(f"\nSelecciona el número del dispositivo de {type_str}: ")
            idx = int(choice) - 1
            if 0 <= idx < len(devices):
                selected_id, selected_name = devices[idx]
                print(f"-> Seleccionado: {selected_name} (ID: {selected_id})")
                return selected_id
            print("Número inválido, intenta nuevamente.")
        except ValueError:
            print("Por favor ingresa un número válido.")

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("INICIANDO SISTEMA PIPECAT")

    # Selección de dispositivos de audio
    p = pyaudio.PyAudio()
    try:
        mic_id = select_device(p, is_input=True)
        spk_id = select_device(p, is_input=False)
    finally:
        p.terminate()

    logger.info(f"Configurando audio (Mic: {mic_id}, Spk: {spk_id})...")

    try:
        transport = LocalAudioTransport(
            params=LocalAudioTransportParams(
                sample_rate=16000,
                audio_out_sample_rate=44100,
                audio_in_enabled=False,
                audio_out_enabled=False,
                audio_in_index=mic_id,
                audio_out_index=spk_id,
                buffer_size=1024
            )
        )
    except Exception as e:
        logger.error(f"ERROR INICIALIZANDO AUDIO: {e}")
        return

    logger.info("Cargando modelos...")
    vad = SileroVADAnalyzer()
    stt = LocalWhisperService(vad_analyzer=vad)
    llm = LocalGemmaService(model="gemma3:12b")
    tts = LocalKokoroService(voice="af_bella", output_sr=16000)
    sentence_aggregator = SentenceAggregator()

    pipeline = Pipeline([
        transport.input(),
        stt,
        llm,
        sentence_aggregator,
        tts,
        transport.output()
    ])

    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    # Iniciar la conversación
    await task.queue_frame(StartFrame())

    async def shutdown(sig, frame):
        logger.info("Apagando agente...")
        await task.cancel()
        sys.exit(0)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(sig, None)))

    logger.info("¡SISTEMA LISTO! Habla ahora (Ctrl+C para salir).")

    try:
        await runner.run(task)
    except asyncio.CancelledError:
        logger.info("Tarea cancelada.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass