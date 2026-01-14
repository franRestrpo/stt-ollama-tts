import asyncio
import logging
import signal
import sys

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


# ¡VERIFICA ESTOS ID CON list_devices.py!
MIC_ID = 6  # HD-Audio Generic: ALC892 Analog
SPK_ID = 6  # HD-Audio Generic: ALC892 Analog

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
    logger.info(f"Configurando audio (Mic: {MIC_ID}, Spk: {SPK_ID})...")

    try:
        transport = LocalAudioTransport(
            params=LocalAudioTransportParams(
                sample_rate=16000,
                audio_out_sample_rate=44100,
                audio_in_enabled=False,
                audio_out_enabled=False,
                audio_in_index=MIC_ID,
                audio_out_index=SPK_ID,
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