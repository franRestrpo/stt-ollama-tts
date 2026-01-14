from pipecat.services.tts_service import TTSService
from pipecat.frames.frames import TextFrame, AudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from kokoro import KPipeline
import numpy as np

class LocalKokoroService(TTSService):
    def __init__(self, voice="af_bella", output_sr=44100):
        super().__init__()
        self._pipeline = KPipeline(lang_code='es')
        self._voice = voice
        self._output_sr = output_sr

    # --- MÉTODO OBLIGATORIO POR LA CLASE PADRE (TTSService) ---
    async def run_tts(self, text: str) -> bytes:
        """
        Método requerido abstractamente. Genera todo el audio de una sola vez.
        Nota: En process_frame usamos streaming, pero debemos tener este definido.
        """
        # Esta implementación es básica y no hace streaming, solo cumple el contrato.
        generator = self._pipeline(text, voice=self._voice, speed=1.1)
        audio_list = []
        for _, _, audio in generator:
            audio_list.append(audio)
            
        if not audio_list:
            return b""
            
        full_audio = np.concatenate(audio_list)
        
        # Resample simple
        num_samples = int(len(full_audio) * self._output_sr / 24000)
        resampled = np.interp(
            np.linspace(0, len(full_audio), num_samples),
            np.arange(len(full_audio)),
            full_audio
        ).astype(np.float32)
        
        return (resampled * 32767).astype(np.int16).tobytes()

    async def process_frame(self, frame, direction):
        if isinstance(frame, TextFrame):
            await self.push_frame(TTSStartedFrame())
            
            try:
                # Generación con streaming (Lo que realmente usamos)
                generator = self._pipeline(frame.text, voice=self._voice, speed=1.1)
                
                for _, _, audio in generator:
                    # RE-MUESTREO: 24kHz -> 44.1kHz
                    num_samples = int(len(audio) * self._output_sr / 24000)
                    resampled = np.interp(
                        np.linspace(0, len(audio), num_samples),
                        np.arange(len(audio)),
                        audio
                    ).astype(np.float32)

                    audio_bytes = (resampled * 32767).astype(np.int16).tobytes()
                    await self.push_frame(AudioRawFrame(audio_bytes, self._output_sr, 1))
            except Exception as e:
                print(f"TTS Error: {e}")

            await self.push_frame(TTSStoppedFrame())
        else:
            await self.push_frame(frame, direction)