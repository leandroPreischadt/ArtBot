import os

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / "environment" / ".env")

DEVICE = os.getenv("DEVICE", "auto")
MODEL_NAME = os.getenv("MODEL_NAME", "")
TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME")
