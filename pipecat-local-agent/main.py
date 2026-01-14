import asyncio
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
# Servicios VAD y Agregadores (CRÍTICOS)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_response import (
    LLMUserContextAggregator, 
    LLMAssistantContextAggregator
)
from pipecat.processors.aggregators.sentence import SentenceAggregator # Para que el TTS no tartamudee

# Tus servicios locales
from services.whisper_stt import LocalWhisperService
from services.gemma_llm import LocalGemmaService
from services.kokoro_tts import LocalKokoroService

async def main():
    # Configuración de audio
    params = LocalAudioTransportParams(
        sample_rate=16000,
        audio_out_sample_rate=44100,
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_index=12, # Verifica que estos ID sigan siendo válidos
        audio_out_index=12,
        buffer_size=1024
    )

    transport = LocalAudioTransport(params=params)

    # Inicialización de servicios
    vad = SileroVADAnalyzer() # Detecta silencio para saber cuándo transcribir
    stt = LocalWhisperService()
    llm = LocalGemmaService(model="gemma3:12b")
    tts = LocalKokoroService(voice="af_bella", output_sr=44100)

    # Contexto y Agregadores
    user_context = LLMUserContextAggregator(context=[
        {"role": "system", "content": "Eres un asistente útil y conciso en español."}
    ])
    assistant_context = LLMAssistantContextAggregator()
    sentence_aggregator = SentenceAggregator() # Agrupa tokens en frases completas

    # Pipeline LÓGICO: Audio -> VAD -> STT -> Contexto -> LLM -> Frases -> TTS -> Salida -> Guardar Contexto
    pipeline = Pipeline([
        transport.input(),    # Entrada Micrófono
        vad,                  # Analiza si hay voz
        stt,                  # Transcribe cuando el VAD dice que paró de hablar
        user_context,         # Agrega historial (User) -> Emite LLMMessagesFrame
        llm,                  # Genera respuesta (Stream) -> Emite TextFrame (chunks)
        sentence_aggregator,  # Junta chunks en oraciones -> Emite TextFrame (frase completa)
        tts,                  # Audio -> Emite AudioRawFrame
        transport.output(),   # Salida Parlantes
        assistant_context     # Guarda lo que dijo el asistente en el historial
    ])

    task = PipelineTask(pipeline)
    runner = PipelineRunner()
    
    print("\n>>> AGENTE PIPECAT (RTX 5060) INICIANDO...")
    print(">>> Esperando voz...")

    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())