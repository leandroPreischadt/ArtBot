from pathlib import Path
import platform

SYSTEM = platform.system()

# paths.py fica em: projeto/src/artbot/config/paths.py
BASE_DIR = Path(__file__).resolve().parents[3]

if SYSTEM in {"Darwin", "Windows", "Linux"}:
    MODELS_DIR = BASE_DIR / "models"
else:
    raise RuntimeError(f"Sistema operacional não suportado: {SYSTEM}")
