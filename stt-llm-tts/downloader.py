from huggingface_hub import list_repo_files, hf_hub_download
import os

def smart_download():
    repo = "hexgrad/Kokoro-82M"
    print(f"--- Explorando contenido de {repo} ---")
    
    try:
        all_files = list_repo_files(repo)
        # Buscamos el modelo ONNX y el archivo de voces sin importar el nombre exacto
        onnx_file = next((f for f in all_files if f.endswith(".onnx")), None)
        voices_file = next((f for f in all_files if f.endswith("voices.bin")), None)
        
        if not onnx_file or not voices_file:
            print("No se encontraron los archivos esperados. Archivos disponibles:")
            print(all_files)
            return

        for f in [onnx_file, voices_file]:
            print(f"Descargando: {f}...")
            hf_hub_download(repo_id=repo, filename=f, local_dir=".")
        
        # Renombramos para que el código principal no sufra
        if onnx_file != "kokoro-v1.0.onnx":
            os.rename(onnx_file, "kokoro-v1.0.onnx")
            print(f"Reasignado {onnx_file} -> kokoro-v1.0.onnx")

    except Exception as e:
        print(f"Error crítico en la exploración: {e}")

if __name__ == "__main__":
    smart_download()