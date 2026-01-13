import os
import urllib.request

def setup_local_models():
    files = {
        "kokoro-v1.0.onnx": "https://github.com/thewhodidthis/kokoro-onnx/releases/download/v0.2.0/kokoro-v1.0.onnx",
        "voices.bin": "https://github.com/thewhodidthis/kokoro-onnx/releases/download/v0.2.0/voices.bin"
    }

    print("--- Verificando modelos locales para la voz ---")
    for name, url in files.items():
        if not os.path.exists(name):
            print(f"Descargando {name}... (esto puede tardar un poco)")
            urllib.request.urlretrieve(url, name)
            print(f"{name} listo.")
        else:
            print(f"{name} ya existe.")

    print("\n--- Asegurando que Gemma 3:12b esté en Ollama ---")
    os.system("ollama pull gemma3:12b")
    print("Entorno listo para usar con tu RTX 5060.")

if __name__ == "__main__":
    setup_local_models()
