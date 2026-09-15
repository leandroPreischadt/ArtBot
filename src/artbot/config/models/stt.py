from faster_whisper import WhisperModel
from artbot.config.settings import DEVICE
import io
import sounddevice as sd
import numpy as np


def load_model():
    
    """variáveis do modelo"""
    model_size = "small"
    device_type = "cpu"
    compute_type = "int8"
    model = WhisperModel(model_size, device=device_type, compute_type=compute_type)
    
    return model

def whisper_model(model): 
    
    """variáveis para gravar o audio"""
    duration = 7.0
    fs = 16000
    number_channels = 1
    
    print("gravando...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=number_channels)
    sd.wait()
    print("Gravação finalizada")
    
    myrecording = myrecording.squeeze()
    myrecording = np.ascontiguousarray(myrecording, dtype=np.float32)

    """inicialização do modelo"""

    segments, info = model.transcribe(myrecording, beam_size=5, vad_filter=False)
            
    text = ""
    
    for segment in segments:
        text += segment.text
    print(text)
    
    return text 