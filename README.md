# Asistente de Voz con IA Local (STT-Ollama-TTS)

Este repositorio contiene un proyecto para construir un asistente de voz que funciona completamente en tu máquina local, utilizando tecnologías de código abierto. El sistema integra Reconocimiento de Voz (STT), un Modelo de Lenguaje Grande (LLM) y Síntesis de Voz (TTS) para crear una experiencia de conversación fluida y privada.

El proyecto principal, **`pipecat-local-agent`**, utiliza el framework `pipecat-ai` para crear un pipeline de datos en tiempo real, ofreciendo una solución robusta y con un rendimiento optimizado.

## 🤖 Demostración Rápida
*(Aquí iría un GIF o video corto mostrando al asistente en acción)*

## ✨ Características Principales

- **100% Local y Privado**: Todas las operaciones (STT, LLM, TTS) se ejecutan en tu propio hardware. Tus conversaciones nunca salen de tu máquina.
- **Componentes de Código Abierto**:
    - **STT**: `whisper` (a través de `pipecat`) para una transcripción rápida y precisa.
    - **LLM**: `Ollama` con el modelo `gemma` para el razonamiento y la generación de respuestas.
    - **TTS**: `kokoro` para una síntesis de voz natural y de alta calidad en español.
- **Pipeline Asíncrono**: Gracias a `pipecat-ai`, el audio se procesa en un flujo continuo, permitiendo interrupciones y una latencia de respuesta muy baja.
- **Selección Interactiva de Dispositivos**: El agente te permite elegir el micrófono y los altavoces al inicio, evitando la necesidad de configurar IDs de dispositivo manualmente.
- **Gestión de Conversación**: Mantiene el historial de la conversación para dar respuestas contextuales.

## 🚀 Cómo Empezar

Esta guía se centra en el proyecto principal y más avanzado: `pipecat-local-agent`.

### 1. Prerrequisitos

- **Hardware**:
    - Una GPU NVIDIA con soporte para CUDA es **muy recomendable** para un rendimiento óptimo, especialmente para el STT (Whisper) y el TTS (Kokoro).
- **Software**:
    - [Python 3.10+](https://www.python.org/downloads/)
    - [Git](https://git-scm.com/downloads)
    - [Ollama](https://ollama.com/) instalado y ejecutándose.
        - Descarga el modelo `gemma`:
          ```bash
          ollama pull gemma
          ```

### 2. Instalación

1.  **Clona el repositorio:**
    ```bash
    git clone https://github.com/tu-usuario/stt-ollama-tts.git
    cd stt-ollama-tts/pipecat-local-agent
    ```

2.  **Crea un entorno virtual:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    # En Windows: .venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    El archivo `requirements.txt` se encarga de instalar `pipecat-ai` y `pyaudio`.
    ```bash
    pip install -r requirements.txt
    ```
    > **Nota sobre PyAudio**: Si encuentras errores durante la instalación de `pyaudio`, puede que necesites instalar las dependencias de desarrollo de PortAudio en tu sistema.
    > - En Debian/Ubuntu: `sudo apt-get install portaudio19-dev`
    > - En Mac (con Homebrew): `brew install portaudio`

### 3. Ejecución

Con tu entorno virtual activado y `ollama` corriendo en segundo plano, ejecuta el agente:

```bash
python main.py
```

Al iniciarse, el programa te pedirá que selecciones el dispositivo de entrada (micrófono) y el de salida (altavoces) de una lista numerada. Simplemente introduce el número correspondiente y presiona Enter.

¡Listo! Habla a tu micrófono y el asistente te responderá.

## 📁 Estructura del Proyecto

```
/pipecat-local-agent
├─── main.py                # Punto de entrada, define y corre el pipeline de pipecat.
├─── requirements.txt       # Dependencias del proyecto.
├─── list_devices.py        # Utilidad para listar dispositivos de audio.
└─── /services
     ├─── gemma_llm.py       # Servicio para interactuar con Ollama/Gemma.
     ├─── kokoro_tts.py      # Servicio para la síntesis de voz con Kokoro.
     └─── whisper_stt.py     # Servicio para la transcripción con Whisper.
```

## alternative Alternativa Simple: `stt-llm-tts`

Dentro del repositorio también encontrarás la carpeta `stt-llm-tts`. Este es un agente de voz mucho más simple, construido con un bucle `while` secuencial en Python y sin usar el framework `pipecat`.

- **Propósito**: Es un excelente recurso educativo para entender el flujo básico de un asistente de voz (Escuchar -> Pensar -> Hablar) de forma lineal.
- **Uso**: Requiere editar manualmente el archivo `stt-llm-tts/main.py` para configurar los IDs de tu micrófono y altavoces. Puedes usar el script `utils/list_devices.py` para encontrarlos.
- **Dependencias**: Tiene una lista de dependencias más explícita en su propio `requirements.txt`.

Es una buena base si quieres experimentar con los componentes individuales antes de pasar a un framework más complejo como `pipecat`.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Si tienes ideas para mejorar el asistente, por favor abre un *issue* o envía un *pull request*.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
