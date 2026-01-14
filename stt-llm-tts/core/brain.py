from ollama import AsyncClient

class Brain:
    def __init__(self, model="gemma3:12b"):
        self.client = AsyncClient()
        self.model = model

    async def think(self, user_input):
        messages = [
            {'role': 'system', 'content': 'Eres un asistente técnico conciso. Responde en español, máximo 2 frases.'},
            {'role': 'user', 'content': user_input}
        ]
        response = await self.client.chat(model=self.model, messages=messages)
        return response['message']['content']