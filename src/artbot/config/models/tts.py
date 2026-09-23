from piper import PiperVoice
import sounddevice as sd
from artbot.config.settings import TTS_MODEL_NAME

# Piper + processamento de áudio -> tentar fazer para uma voz mais robótica 

def load_piper_model():
    voice = PiperVoice.load(TTS_MODEL_NAME)
    return voice

def piper_model(voice, text):
    with sd.RawOutputStream(
    samplerate=voice.config.sample_rate,
    channels=1,
    dtype="int16",) as stream:
        for chunk in voice.synthesize(text):
            stream.write(chunk.audio_int16_bytes)