import os

from dotenv import load_dotenv

load_dotenv()

DEVICE = os.getenv("DEVICE", "auto")
MODEL_NAME = os.getenv("MODEL_NAME", "")
