# 🎙️ Local Voice AI Agent (Gemma 3 + Faster-Whisper + Kokoro)

Este proyecto implementa un asistente de voz 100% local y privado, optimizado para la arquitectura NVIDIA Ada Lovelace (**RTX 5060**). El sistema integra transcripción en tiempo real, un modelo de lenguaje de última generación y síntesis de voz con calidad humana.

## 🏗️ Arquitectura del Sistema

El flujo de datos se procesa íntegramente en la GPU para minimizar la latencia:
1.  **Escucha (STT):** [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (Modelo `small`, `float16`).
2.  **Cerebro (LLM):** [Gemma 3 12B](https://ollama.com/library/gemma3) vía Ollama.
3.  **Habla (TTS):** [Kokoro v0.9.4+](https://github.com/hexgrad/kokoro) para una entonación natural.



---

## 🛠️ Requisitos e Instalación

### 1. Requisitos del Sistema
* **GPU:** NVIDIA RTX 5060 (16GB VRAM recomendados).
* **SO:** Linux (Debian/Ubuntu/Arch) con drivers NVIDIA y CUDA configurados.
* **Dependencia Externa:** [Ollama](https://ollama.ai/) debe estar instalado y el servicio activo.

### 2. Configuración del Entorno
```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias del sistema para audio
sudo apt update && sudo apt install libportaudio2 libasound2-dev

# Instalar librerías de Python
pip install -r requirements.txt