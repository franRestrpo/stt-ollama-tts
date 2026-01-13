import asyncio
from ollama import AsyncClient

async def test_brain():
    try:
        print("Consultando a Gemma 3 en la GPU...")
        response = await AsyncClient().chat(model='gemma3:12b', messages=[
            {'role': 'user', 'content': 'Hola, ¿estás funcionando en mi RTX 5060?'}
        ])
        print("Respuesta de la IA:", response['message']['content'])
    except Exception as e:
        print(f"Error de conexión con Ollama: {e}")

if __name__ == "__main__":
    asyncio.run(test_brain())