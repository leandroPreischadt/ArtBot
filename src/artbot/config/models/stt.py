from faster_whisper import WhisperModel
from artbot.config.settings import (
    DEVICE,
    AUDIO_DEVICE,
    VAD_AGGRESSIVENESS,
    START_FRAMES,
    END_SILENCE_FRAMES,
    NUMBER_OF_CHANNELS
)
import sounddevice as sd
import numpy as np
import queue
import collections
import webrtcvad

# Variáveis do Hardware (DJI Mic)
SAMPLE_RATE_HW = 48000     
CHANNELS_HW = NUMBER_OF_CHANNELS
FRAME_MS = 30
FRAME_SAMPLES_HW = int(SAMPLE_RATE_HW * FRAME_MS / 1000) # 1440 amostras

# Variáveis para a IA (Whisper/VAD)
SAMPLE_RATE_AI = 16000
FRAME_SAMPLES_AI = int(SAMPLE_RATE_AI * FRAME_MS / 1000) # 480 amostras

audio_queue = queue.Queue()

def load_stt_model():
    """Models vars"""
    model_size = "small"
    device_type = DEVICE
    compute_type = "int8_float16" if device_type == "cuda" else "int8"
    
    """initiating model"""
    # Corrigido para habilitar a GPU (Jetson)
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
    
    return model

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio input status: {status}")

    # O DJI entrega estéreo em 48 kHz. O Whisper/VAD usa mono em 16 kHz.
    # Os dois canais do receptor são somados para não depender de qual
    # transmissor está associado ao canal esquerdo.
    mono = indata.mean(axis=1).astype(np.int16)

    # Downsample 48 kHz -> 16 kHz.
    audio_queue.put(mono[::3].copy().tobytes())
    
def record_until_silence():
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    buffer = bytearray()
    
    pre_roll = collections.deque(maxlen=10)
    
    speech_started = False
    voiced_frames = 0
    silent_frames = 0
    
    print("Waiting voice...")
    
    with sd.InputStream(
        device=AUDIO_DEVICE,
        samplerate=SAMPLE_RATE_HW,
        channels=CHANNELS_HW,
        dtype="int16",
        blocksize=FRAME_SAMPLES_HW,
        callback=audio_callback,
    ):
         while True:
            frame = audio_queue.get()

            # Como o frame já foi convertido no callback, esperamos os bytes de 16kHz
            expected_bytes = FRAME_SAMPLES_AI * 2
            
            if len(frame) != expected_bytes:
                continue
            
            # O VAD analisa o áudio convertido a 16000 Hz
            is_speech = vad.is_speech(frame, SAMPLE_RATE_AI)
            
            if not speech_started:
                pre_roll.append(frame)

                if is_speech:
                    voiced_frames += 1
                else:
                    voiced_frames = 0

                if voiced_frames >= START_FRAMES:
                    speech_started = True
                    
                    buffer.extend(b"".join(pre_roll))
                    silent_frames = 0

                    print("Voice detected...")
            else:
                buffer.extend(frame)

                if is_speech:
                    silent_frames = 0
                else:
                    silent_frames += 1

                if silent_frames >= END_SILENCE_FRAMES:
                    print("Recorded voice.")
                    break
                    
    audio = np.frombuffer(buffer, dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0 # type: ignore

    return audio

def whisper_model(model): 
    audio = record_until_silence()
    
    segments, info = model.transcribe(audio, beam_size=5, vad_filter=False)
            
    text = "".join(segment.text for segment in segments).strip()

    print(text)
    return text
