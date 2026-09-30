import os

from pathlib import Path

from dotenv import load_dotenv

from artbot.config.paths import BASE_DIR

load_dotenv(Path(__file__).resolve().parents[3] / "environment" / ".env")

DEVICE = os.getenv("DEVICE", "auto")
MODEL_NAME = os.getenv("MODEL_NAME", "")
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "")
N_CTX = int(os.getenv("N_CTX", ""))

# O PortAudio usa ``None`` para o dispositivo padrão do sistema.  ``default``
# é um alias útil no .env, mas não é necessariamente um nome de dispositivo
# válido no macOS.
_audio_device = os.getenv("AUDIO_DEVICE", "pipewire").strip()
AUDIO_DEVICE = (
    None if _audio_device.lower() in {"", "default", "system", "auto"}
    else _audio_device
)

VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "1"))
START_FRAMES = int(os.getenv("START_FRAMES", "3"))
END_SILENCE_FRAMES = int(os.getenv("END_SILENCE_FRAMES", "40"))
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))

HTTP_PORT = os.getenv("HTTP_PORT", "")

_database_path = Path(os.getenv("DATABASE_PATH") or BASE_DIR / "data" / "chroma")
DATABASE_PATH = str(_database_path if _database_path.is_absolute() else BASE_DIR / _database_path)

KNOWLEDGE_PATH = BASE_DIR / "knowledge.json"
NUMBER_OF_CHANNELS = int(os.getenv("NUMBER_OF_CHANNELS", "2"))
