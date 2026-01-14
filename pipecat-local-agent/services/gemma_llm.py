from pipecat.services.llm_service import LLMService
from pipecat.frames.frames import LLMMessagesFrame, TextFrame, LLMFullResponseEndFrame
from ollama import AsyncClient

class LocalGemmaService(LLMService):
    def __init__(self, model="gemma3:12b"):
        super().__init__()
        self._model = model
        self._client = AsyncClient()

    async def process_frame(self, frame, direction):
        # AHORA escuchamos LLMMessagesFrame, no TextFrame
        if isinstance(frame, LLMMessagesFrame):
            print(f"LLM Processing messages: {len(frame.messages)}")
            
            try:
                # Streaming real con el historial completo
                async for chunk in self._client.chat(
                    model=self._model,
                    messages=frame.messages, # Pasamos el historial, no solo el último texto
                    stream=True
                ):
                    content = chunk['message']['content']
                    if content:
                        await self.push_frame(TextFrame(content))
                
                await self.push_frame(LLMFullResponseEndFrame())
            except Exception as e:
                print(f"Error LLM: {e}")
        else:
            await self.push_frame(frame, direction)