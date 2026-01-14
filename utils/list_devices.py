import sounddevice as sd

def list_devices():
    print(f"\n{'ID':<4} {'NOMBRE':<40} {'CANALES (IN/OUT)':<20}")
    print("-" * 70)
    
    # Listamos todos los dispositivos
    devices = sd.query_devices()
    
    for i, dev in enumerate(devices):
        # Filtramos un poco para limpiar la vista (opcional)
        name = dev['name'][:38]
        inputs = dev['max_input_channels']
        outputs = dev['max_output_channels']
        
        # Marcamos con flechas lo que parece útil
        mark = ""
        if "USB" in name or "HDA" in name: 
            mark = " <--- REVISAR ESTE"
            
        if inputs > 0 or outputs > 0:
            print(f"{i:<4} {name:<40} {inputs}/{outputs:<20} {mark}")

if __name__ == "__main__":
    list_devices()