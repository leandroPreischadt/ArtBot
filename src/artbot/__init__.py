from artbot.config.paths import MODELS_DIR
from artbot.config.settings import DEVICE, MODEL_NAME
from artbot.config.models.stt import whisper_model

MODEL_PATH = MODELS_DIR / MODEL_NAME


def main() -> None:
    print(f"MODEL: {MODEL_PATH}")
    print(f"DEVICE: {DEVICE}")
    whisper_model()
    


if __name__ == "__main__":
    main()
