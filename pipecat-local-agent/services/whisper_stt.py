from pipecat.services.stt_service import STTService
from pipecat.frames.frames import AudioRawFrame, TextFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from faster_whisper import WhisperModel
import numpy as np

class LocalWhisperService(STTService):
    def __init__(self):
        super().__init__()
        # Optimizamos para RTX 5060 (16GB)
        self._model = WhisperModel("medium", device="cuda", compute_type="float16")
        self._audio_buffer = bytearray()
        self._is_recording = False

    # --- MÉTODO OBLIGATORIO POR LA CLASE PADRE (STTService) ---
    async def run_stt(self, audio: bytes) -> str:
        """
        Este método es requerido por STTService. 
        Toma bytes de audio y devuelve el texto transcrito.
        """
        if not audio:
            return ""
            
        # Conversión a float32 normalizado (requerido por Faster-Whisper)
        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcripción (bloqueante en GPU, idealmente iría en un thread aparte, pero funciona rápido en 5060)
        segments, _ = self._model.transcribe(audio_np, language="es")
        text = " ".join([s.text for s in segments]).strip()
        return text

    async def process_frame(self, frame, direction):
        """
        Manejo manual de frames para acumular audio basado en VAD.
        """
        # 1. Inicio de voz detectado por Silero
        if isinstance(frame, UserStartedSpeakingFrame):
            self._is_recording = True
            self._audio_buffer = bytearray()
            # No enviamos este frame hacia abajo para no confundir al LLM todavía, 
            # o podemos dejarlo pasar si queremos feedback visual en UI.

        # 2. Fin de voz detectado
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._is_recording = False
            if len(self._audio_buffer) > 0:
                # LLAMAMOS AL MÉTODO OBLIGATORIO AQUÍ
                text = await self.run_stt(bytes(self._audio_buffer))
                if text:
                    print(f"User (Whisper): {text}")
                    await self.push_frame(TextFrame(text))
        
        # 3. Audio crudo llegando del micrófono
        elif isinstance(frame, AudioRawFrame):
            if self._is_recording:
                self._audio_buffer.extend(frame.audio)
        
        # 4. Otros frames (System, etc.)
        else:
            await self.push_frame(frame, direction)