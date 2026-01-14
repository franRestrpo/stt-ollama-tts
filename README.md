# Local Voice AI Project

This repository hosts implementations of local, privacy-focused Voice AI agents. It is designed to run entirely on local hardware (optimized for NVIDIA GPUs) without relying on external cloud APIs for transcription, intelligence, or synthesis.

## internal modules

### 1. [stt-llm-tts](./stt-llm-tts)

A complete voice assistant pipeline integrating:

- **STT (Ears):** [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- **LLM (Brain):** [Gemma 3](https://ollama.com/library/gemma3) (via Ollama)
- **TTS (Voice):** [Kokoro](https://github.com/hexgrad/kokoro)

See [stt-llm-tts/README.md](./stt-llm-tts/README.md) for detailed setup and usage.

### 2. [pipecat-local-agent](./pipecat-local-agent)

An agent implementation building upon the [Pipecat](https://github.com/pipecat-ai/pipecat) framework, designed for modular and scalable real-time conversational AI.

## General Prerequisites

- **OS:** Linux (Debian/Ubuntu/Arch recommended)
- **GPU:** NVIDIA GPU with CUDA support (RTX 3060/4060/5060 or better recommended)
- **Python:** 3.10+
- **External Tools:**
  - [Ollama](https://ollama.ai/) (for running local LLMs)
  - System audio libraries (`portaudio`, `alsa`)

## Quick Start

Each module manages its own dependencies. Navigate to the specific directory of the agent you wish to run and follow its internal README.

```bash
# Example for stt-llm-tts
cd stt-llm-tts
pip install -r requirements.txt
python main.py
```
