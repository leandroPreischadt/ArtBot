from piper import PiperVoice
import sounddevice as sd
import numpy as np
from artbot.config.settings import TTS_MODEL_NAME

# Piper + processamento de áudio -> tentar fazer para uma voz mais robótica 


def load_piper_model():
    voice = PiperVoice.load(TTS_MODEL_NAME)
    return voice

def piper_model(voice, text):
    audio_chunks = []

    for chunk in voice.synthesize(text["choices"][0]["message"]["content"]): # type: ignore
        audio_chunks.append(chunk.audio_int16_bytes)

    audio_bytes = b"".join(audio_chunks)
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

    sd.play(audio_array, samplerate=voice.config.sample_rate)
    sd.wait()