
import asyncio
import logging
import sys
import pyaudio
import signal
import os

# Add the current directory to sys.path so we can import services
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.frames.frames import StartFrame, TextFrame
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.frame_processor import FrameProcessor

# Import the existing Whisper Service (which has the fixes)
from services.whisper_stt import LocalWhisperService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextToFileWriter(FrameProcessor):
    def __init__(self, filename="transcription_output.txt"):
        super().__init__()
        self.filename = filename
        # Initialize file with header
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write("--- SESIÓN DE TRANSCRIPCIÓN ---\n")
        print(f"📁 Guardando transcripción en: {os.path.abspath(self.filename)}")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            text = frame.text
            print(f"📝 Detectado: {text}")
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(f"{text}\n")
        
        await self.push_frame(frame, direction)

def select_input_device(p):
    """Select input device helper."""
    count = p.get_device_count()
    devices = []
    print("\n--- INPUT DEVICES ---")
    idx_counter = 1
    for i in range(count):
        dev = p.get_device_info_by_index(i)
        if dev['maxInputChannels'] > 0:
            devices.append((i, dev['name']))
            print(f"{idx_counter}. {dev['name']}")
            idx_counter += 1
            
    if not devices:
        print("Error: No input devices found.")
        sys.exit(1)

    while True:
        try:
            choice = input(f"\nSelect input device number: ")
            idx = int(choice) - 1
            if 0 <= idx < len(devices):
                selected_id, selected_name = devices[idx]
                print(f"-> Selected: {selected_name} (ID: {selected_id})")
                return selected_id
            print("Invalid number.")
        except ValueError:
            print("Enter a valid number.")

async def main():
    logger.info("Iniciando prueba de STT (Voz a Texto)...")
    
    p = pyaudio.PyAudio()
    try:
        mic_id = select_input_device(p)
    finally:
        p.terminate()

    # 1. Transport
    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            sample_rate=16000,
            audio_out_enabled=False,
            audio_in_enabled=True,
            audio_in_index=mic_id,
            buffer_size=1024
        )
    )

    # 2. VAD (Voice Activity Detection)
    vad = SileroVADAnalyzer()

    # 3. STT (Speech to Text)
    # Using the LocalWhisperService which we know has the proper configuration
    stt = LocalWhisperService(vad_analyzer=vad)

    # 4. File Writer
    file_writer = TextToFileWriter("transcription_output.txt")

    pipeline = Pipeline([
        transport.input(),
        stt,
        file_writer
    ])

    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    await task.queue_frame(StartFrame())

    async def shutdown(sig, frame):
        print("\nFinalizando...")
        await task.cancel()
        sys.exit(0)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(sig, None)))

    print("\n--- PRUEBA INICIADA ---")
    print("Habla ahora. El texto aparecerá aquí y en 'transcription_output.txt'")
    print("Presiona Ctrl+C para salir.\n")

    try:
        await runner.run(task)
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
