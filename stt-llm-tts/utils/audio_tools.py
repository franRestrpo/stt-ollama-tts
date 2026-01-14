import numpy as np

def resample_audio(audio, source_sr, target_sr):
    """Re-muestreo lineal eficiente para compatibilidad de hardware."""
    num_samples = int(len(audio) * target_sr / source_sr)
    return np.interp(
        np.linspace(0, len(audio), num_samples), 
        np.arange(len(audio)), 
        audio
    ).astype(np.float32)