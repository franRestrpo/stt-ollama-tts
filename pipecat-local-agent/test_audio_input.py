
import asyncio
import logging
import sys
import pyaudio
import signal
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.frames.frames import StartFrame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class AudioVolumeLogger(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            # Calculate simple volume/energy to verify signal
            import numpy as np
            audio_data = np.frombuffer(frame.audio, dtype=np.int16)
            volume = np.linalg.norm(audio_data) / len(audio_data)
            # Log every few frames or if volume is significant
            if volume > 0.1: 
                print(f"🎤 Audio detected! Volume: {volume:.2f} | Bytes: {len(frame.audio)}")
            else:
                 # Print a dot for silence to know it's running
                 print(".", end="", flush=True)
        
        await self.push_frame(frame, direction)

async def main():
    logger.info("Starting Audio Input Validation...")
    
    p = pyaudio.PyAudio()
    try:
        mic_id = select_input_device(p)
    finally:
        p.terminate()

    transport = LocalAudioTransport(
        params=LocalAudioTransportParams(
            sample_rate=16000,
            audio_out_enabled=False,
            audio_in_enabled=True,
            audio_in_index=mic_id,
            buffer_size=1024 # Smaller buffer for faster feedback
        )
    )

    volume_logger = AudioVolumeLogger()

    pipeline = Pipeline([
        transport.input(),
        volume_logger
    ])

    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    await task.queue_frame(StartFrame())

    async def shutdown(sig, frame):
        print("\nShutting down...")
        await task.cancel()
        sys.exit(0)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(sig, None)))

    print("\n--- TEST STARTED ---")
    print("Speak into your microphone. You should see volume logs.")
    print("If you only see dots '...', the mic is capturing silence.")
    print("Press Ctrl+C to stop.\n")

    try:
        await runner.run(task)
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
