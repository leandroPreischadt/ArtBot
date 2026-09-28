import os

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / "environment" / ".env")

DEVICE = os.getenv("DEVICE", "auto")
MODEL_NAME = os.getenv("MODEL_NAME", "")
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME")
N_CTX = int(os.getenv("N_CTX", ""))

VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "1"))
START_FRAMES = int(os.getenv("START_FRAMES", "3"))
END_SILENCE_FRAMES = int(os.getenv("END_SILENCE_FRAMES", "40"))
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))

HTTP_PORT = os.getenv("HTTP_PORT", "")
