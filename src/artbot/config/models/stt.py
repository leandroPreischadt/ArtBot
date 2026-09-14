from faster_whisper import WhisperModel
from artbot.config.settings import DEVICE

def whisper_model():
    # Initialize model (use device="cuda" if you have a compatible GPU)
    model = WhisperModel("small", device=DEVICE, compute_type="int8")

    # Transcribe audio file
    segments, info = model.transcribe("audio.mp3", beam_size=5)

    print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
