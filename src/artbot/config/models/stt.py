from faster_whisper import WhisperModel
import ctranslate2
from artbot.config.settings import (
    DEVICE,
    AUDIO_DEVICE,
    VAD_AGGRESSIVENESS,
    START_FRAMES,
    END_SILENCE_FRAMES,
    NUMBER_OF_CHANNELS,
    STT_BEAM_SIZE,
    STT_LANGUAGE,
    STT_MODEL_SIZE,
    STT_MAX_RECORDING_SECONDS,
)
import sounddevice as sd
import numpy as np
import queue
import collections
import webrtcvad
import logging

# Variáveis do Hardware (DJI Mic)
SAMPLE_RATE_HW = 48000     
CHANNELS_HW = NUMBER_OF_CHANNELS
FRAME_MS = 30
FRAME_SAMPLES_HW = int(SAMPLE_RATE_HW * FRAME_MS / 1000) # 1440 amostras

# Variáveis para a IA (Whisper/VAD)
SAMPLE_RATE_AI = 16000
FRAME_SAMPLES_AI = int(SAMPLE_RATE_AI * FRAME_MS / 1000) # 480 amostras

logger = logging.getLogger(__name__)

# Evita que uma falha/pausa do consumidor faça a RAM crescer indefinidamente.
audio_queue = queue.Queue(maxsize=64)

def load_stt_model():
    requested_cuda = DEVICE == "cuda"
    try:
        cuda_compute_types = ctranslate2.get_supported_compute_types("cuda")
    except (RuntimeError, ValueError):
        cuda_compute_types = set()

    if requested_cuda and cuda_compute_types:
        device_type = "cuda"
        compute_type = "int8_float16" if "int8_float16" in cuda_compute_types else "float16"
    else:
        device_type = "cpu"
        compute_type = "int8"
        if requested_cuda:
            print(
                "STT: CTranslate2 sem suporte CUDA neste ambiente; "
                "usando CPU. Instale uma build CTranslate2 CUDA para acelerar o Whisper."
            )

    print(f"STT backend: {device_type} | compute_type={compute_type}")
    return WhisperModel(
        STT_MODEL_SIZE,
        device=device_type,
        compute_type=compute_type,
    )

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio input status: {status}")

    # O DJI entrega estéreo em 48 kHz. O Whisper/VAD usa mono em 16 kHz.
    # Os dois canais do receptor são somados para não depender de qual
    # transmissor está associado ao canal esquerdo.
    mono = indata.mean(axis=1).astype(np.int16)

    # Downsample 48 kHz -> 16 kHz.
    data = mono[::3].copy().tobytes()
    try:
        audio_queue.put_nowait(data)
    except queue.Full:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            audio_queue.put_nowait(data)
        except queue.Full:
            logger.warning("Fila de áudio STT cheia; descartando frame")
    
def record_until_silence():
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    buffer = bytearray()
    
    pre_roll = collections.deque(maxlen=10)
    
    speech_started = False
    voiced_frames = 0
    silent_frames = 0
    max_frames = max(1, int(STT_MAX_RECORDING_SECONDS * 1000 / FRAME_MS))
    received_frames = 0
    
    print("Waiting voice...")
    
    with sd.InputStream(
        device=AUDIO_DEVICE,
        samplerate=SAMPLE_RATE_HW,
        channels=CHANNELS_HW,
        dtype="int16",
        blocksize=FRAME_SAMPLES_HW,
        callback=audio_callback,
    ):
         while received_frames < max_frames:
            frame = audio_queue.get()
            received_frames += 1

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

    if not speech_started:
        print("Nenhuma voz detectada dentro do limite de gravação.")
        return np.empty(0, dtype=np.float32)
                    
    audio = np.frombuffer(buffer, dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0 # type: ignore

    return audio

def whisper_model(model): 
    audio = record_until_silence()

    if audio.size == 0:
        return ""
    
    segments, info = model.transcribe(
        audio,
        language=STT_LANGUAGE,
        beam_size=STT_BEAM_SIZE,
        vad_filter=False,
        condition_on_previous_text=False,
    )
            
    text = "".join(segment.text for segment in segments).strip()

    print(text)
    return text
