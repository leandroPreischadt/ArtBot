import os

from pathlib import Path

from dotenv import load_dotenv

from artbot.config.paths import BASE_DIR

load_dotenv(Path(__file__).resolve().parents[3] / "environment" / ".env")

DEVICE = os.getenv("DEVICE", "auto")
MODEL_NAME = os.getenv("MODEL_NAME", "")
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "")
N_CTX = int(os.getenv("N_CTX", ""))

# O PortAudio enxerga o DJI Mic Mini através do nó PipeWire.  Na Jetson,
# ``default`` aponta para a entrada APE e não para o microfone USB.
AUDIO_DEVICE = os.getenv("AUDIO_DEVICE", "pipewire")

VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "1"))
START_FRAMES = int(os.getenv("START_FRAMES", "3"))
END_SILENCE_FRAMES = int(os.getenv("END_SILENCE_FRAMES", "40"))
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))

HTTP_PORT = os.getenv("HTTP_PORT", "")

_database_path = Path(os.getenv("DATABASE_PATH") or BASE_DIR / "data" / "chroma")
DATABASE_PATH = str(_database_path if _database_path.is_absolute() else BASE_DIR / _database_path)

KNOWLEDGE_PATH = BASE_DIR / "knowledge.json"
