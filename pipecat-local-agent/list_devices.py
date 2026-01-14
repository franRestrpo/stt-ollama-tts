import pyaudio

p = pyaudio.PyAudio()

print("Dispositivos de entrada:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"Índice {i}: {info['name']}")

print("\nDispositivos de salida:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxOutputChannels'] > 0:
        print(f"Índice {i}: {info['name']}")

p.terminate()