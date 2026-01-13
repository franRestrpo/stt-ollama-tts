import sounddevice as sd
from kokoro import KPipeline
from utils.audio_tools import resample_audio

class Speaker:
    def __init__(self, output_sr=44100):
        self.pipeline = KPipeline(lang_code='es')
        self.output_sr = output_sr

    async def speak(self, text):
        generator = self.pipeline(text, voice='em_alex', speed=1.1)
        for _, _, audio in generator:
            # Aplicamos el resampling antes de enviar al hardware USB
            final_audio = resample_audio(audio, 24000, self.output_sr)
            sd.play(final_audio, self.output_sr)
            sd.wait()