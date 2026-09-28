import os

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / "environment" / ".env")

HTTP_PORT = int(os.getenv("HTTP_PORT", 8383))
