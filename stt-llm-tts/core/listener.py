import asyncio
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

class Listener:
    def __init__(self, model_size="medium", device="cuda"):
        self.model = WhisperModel(model_size, device=device, compute_type="float16")
        self.sample_rate = 16000
        self.energy_threshold = 0.5
        self.silence_limit = 0.7  # Segundos de silencio para cortar

    async def listen(self):
        audio_buffer = []
        recording = False
        silence_counter = 0

        # Log device information for debugging
        print("Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"Device {i}: {device['name']}, Input channels: {device['max_input_channels']}, Output channels: {device['max_output_channels']}, Default sample rate: {device['default_samplerate']}")
        print(f"Default input device: {sd.default.device['input']}")
        print(f"Default output device: {sd.default.device['output']}")

        def callback(indata, frames, time, status):
            nonlocal recording, silence_counter
            volume = np.linalg.norm(indata) * 10
            if volume > self.energy_threshold:
                recording = True
                silence_counter = 0
                audio_buffer.append(indata.copy())
            elif recording:
                silence_counter += frames / self.sample_rate
                audio_buffer.append(indata.copy())

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=callback):
                while not recording or silence_counter < self.silence_limit:
                    await asyncio.sleep(0.1)
        except sd.PortAudioError as e:
            print(f"Error opening InputStream: {e}")
            raise
            while not recording or silence_counter < self.silence_limit:
                await asyncio.sleep(0.1)
        
        return np.concatenate(audio_buffer).flatten()

    async def transcribe(self, audio_data):
        segments, info = self.model.transcribe(audio_data, language="es", vad_filter=True)
        text = " ".join([s.text for s in segments]).strip()
        # Filtro de seguridad contra alucinaciones de Whisper
        if info.language_probability < 0.5 or len(text) < 3:
            return ""
        return text