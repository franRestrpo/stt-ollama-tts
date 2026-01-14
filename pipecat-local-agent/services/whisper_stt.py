import logging

from pipecat.services.stt_service import STTService
from pipecat.frames.frames import TextFrame
from faster_whisper import WhisperModel
import numpy as np
import asyncio

logger = logging.getLogger(__name__)

class LocalWhisperService(STTService):
    def __init__(self, vad_analyzer=None):
        super().__init__(vad_analyzer=vad_analyzer)
        # Optimizamos para RTX 5060 (16GB)
        self._model = WhisperModel("medium", device="cuda", compute_type="float16")

    async def run_stt(self, audio: bytes):
        """
        Este método es requerido por STTService.
        Toma bytes de audio y devuelve un async iterable de frames (típicamente TextFrame).
        """
        if not audio:
            return

        # Conversión a float32 normalizado (requerido por Faster-Whisper)
        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcripción (bloqueante en GPU, idealmente iría en un thread aparte, pero funciona rápido en 5060)
        segments, _ = await asyncio.get_event_loop().run_in_executor(None, self._model.transcribe, audio_np, "es")
        text = " ".join([s.text for s in segments]).strip()
        if text:
            logger.info(f"User (Whisper): {text}")
            yield TextFrame(text)

    async def process_frame(self, frame, direction):
        """
        Procesar frames para consumir InputAudioRawFrame y pasar otros.
        """
        await super().process_frame(frame, direction)

        from pipecat.frames.frames import InputAudioRawFrame
        if isinstance(frame, InputAudioRawFrame):
            # Consumir el frame de audio sin pasar
            pass