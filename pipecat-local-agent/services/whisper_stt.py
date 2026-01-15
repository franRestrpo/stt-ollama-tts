import logging
import functools

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
        transcribe_func = functools.partial(self._model.transcribe, vad_filter=True)
        segments, info = await asyncio.get_event_loop().run_in_executor(None, transcribe_func, audio_np, "es")
        text = " ".join([s.text for s in segments]).strip()

        # Filtros anti-alucinación (mismos parámetros que stt-llm-tts)
        if info.language_probability < 0.5:
            if text:
                logger.info(f"Whisper ignorado (Baja prob {info.language_probability:.2f}): {text}")
            return

        # Lista negra de alucinaciones comunes de Whisper en silencio
        HALLUCINATIONS = [
            "subtítulos por la comunidad de amara.org",
            "¡gracias por ver el vídeo!",
            "gracias por ver el video",
            "suscríbete",
            "¡suscríbete!",
            "amara.org",
            "subtítulos realizados por",
            "transcripción realizada por"
        ]

        text_lower = text.lower()
        if any(h in text_lower for h in HALLUCINATIONS):
             logger.info(f"Whisper ignorado (Alucinación conocida): {text}")
             return

        if len(text) < 3:
             if text:
                 logger.info(f"Whisper ignorado (Muy corto): {text}")
             return


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