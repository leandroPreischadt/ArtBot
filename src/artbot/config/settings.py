import os

from pathlib import Path

from dotenv import load_dotenv

from artbot.config.paths import BASE_DIR

load_dotenv(Path(__file__).resolve().parents[3] / "environment" / ".env")

# Keep the value used by the model runtimes consistent.  ``gpu`` was used in
# the original .env, while CTranslate2 and llama.cpp expect ``cuda``.
_device = os.getenv("DEVICE", "auto").strip().lower()
DEVICE = {
    "gpu": "cuda",
    "cuda:0": "cuda",
    "nvidia": "cuda",
    "host": "cpu",
}.get(_device, _device)
if DEVICE not in {"auto", "cpu", "cuda"}:
    raise ValueError("DEVICE deve ser auto, cpu ou cuda (gpu também é aceito)")

MODEL_NAME = os.getenv("MODEL_NAME", "")
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "")


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in {None, ""} else int(value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in {None, ""}:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# llama.cpp defaults to n_gpu_layers=0.  That default is the main reason a
# CUDA-capable Jetson can still run the LLM entirely on the CPU.
N_CTX = _int_env("N_CTX", 1024)
MAX_TOKENS = _int_env("MAX_TOKENS", 3000)
N_GPU_LAYERS = _int_env("N_GPU_LAYERS", -1 if DEVICE == "cuda" else 0)
# Small values are intentional: the Orin shares its 8 GB between CPU, GPU,
# the desktop/face process, Whisper and llama.cpp. Larger batches increase
# temporary llama.cpp allocations enough to trigger the kernel OOM killer.
N_BATCH = _int_env("N_BATCH", 16)
N_UBATCH = _int_env("N_UBATCH", 16)
N_THREADS = _int_env("N_THREADS", 0)
N_THREADS_BATCH = _int_env("N_THREADS_BATCH", 0)
FLASH_ATTN = _bool_env("FLASH_ATTN", DEVICE == "cuda")
OFFLOAD_KQV = _bool_env("OFFLOAD_KQV", True)

# STT tuning.  beam_size=1 is substantially faster for short assistant
# commands and can be raised in .env when transcription quality is preferred.
STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "base")
STT_BEAM_SIZE = _int_env("STT_BEAM_SIZE", 1)
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "pt")
STT_MAX_RECORDING_SECONDS = _int_env("STT_MAX_RECORDING_SECONDS", 30)

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

HTTP_PORT = _int_env("HTTP_PORT", 8383)

_database_path = Path(os.getenv("DATABASE_PATH") or BASE_DIR / "data" / "chroma")
DATABASE_PATH = str(_database_path if _database_path.is_absolute() else BASE_DIR / _database_path)

KNOWLEDGE_PATH = BASE_DIR / "knowledge.json"
NUMBER_OF_CHANNELS = int(os.getenv("NUMBER_OF_CHANNELS", "2"))
