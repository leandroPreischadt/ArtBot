from faster_whisper import WhisperModel
from artbot.config.settings import DEVICE
import sounddevice as sd
import numpy as np
import queue
import collections
import webrtcvad

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

audio_queue = queue.Queue()

def load_stt_model():
    
    """Models vars"""
    model_size = "small"
    device_type = "cpu"
    compute_type = "int8"
    
    """initiating model"""
    model = WhisperModel(model_size, device=device_type, compute_type=compute_type)
    
    return model

def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy().tobytes())
    
def record_until_silance():
    vad = webrtcvad.Vad(2)
    buffer = bytearray()
    
    pre_roll = collections.deque(maxlen=10)
    
    speech_started = False
    voiced_frames = 0
    silent_frames = 0
    
    START_FRAMES = 3
    END_SILENCE_FRAMES = 15
    
    print("Waiting voice...")
    
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=FRAME_SAMPLES,
        callback=audio_callback,
    ):
         while True:
            frame = audio_queue.get()

            expected_bytes = FRAME_SAMPLES * 2
            
            if len(frame) != expected_bytes:
                continue
            
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            
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
    audio = audio.astype(np.float32) / 32768.0

    return audio

def whisper_model(model): 
    
    audio = record_until_silance()
    
    segments, info = model.transcribe(audio, beam_size=5, vad_filter= False)
            
    text = "".join(segment.text for segment in segments).strip()

    print(text)
    return text