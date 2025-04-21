import numpy as np
import soundfile as sf
import os

def save_debug_audio(np_audio, rate=16000, tag="test"):
    """
    Save a numpy audio array as a .wav file in a debug directory for inspection.
    """
    debug_dir = "debug_audio_dump"
    os.makedirs(debug_dir, exist_ok=True)
    filename = os.path.join(debug_dir, f"debug_{tag}.wav")
    # Ensure int16 format for wav
    if np_audio.dtype != np.int16:
        np_audio = (np_audio * 32767).astype(np.int16)
    sf.write(filename, np_audio, rate)
    print(f"[DEBUG] Saved audio to {filename}")
    return filename
