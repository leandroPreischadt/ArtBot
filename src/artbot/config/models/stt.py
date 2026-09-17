from faster_whisper import WhisperModel
from artbot.config.settings import DEVICE
import io
import sounddevice as sd
import numpy as np


def load_stt_model():
    
    """Models vars"""
    model_size = "small"
    device_type = "cpu"
    compute_type = "int8"
    
    """initiating model"""
    model = WhisperModel(model_size, device=device_type, compute_type=compute_type)
    
    return model

def whisper_model(model): 
    
    """Recording Vars"""
    duration = 7.0
    fs = 16000
    number_channels = 1
    
    print("Recording...")
    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=number_channels)
    sd.wait()
    print("Recording finished")
    
    myrecording = myrecording.squeeze()
    myrecording = np.ascontiguousarray(myrecording, dtype=np.float32)


    segments, info = model.transcribe(myrecording, beam_size=5, vad_filter=False)
            
    text = ""
    
    for segment in segments:
        text += segment.text
    print(text)
    
    return text 
