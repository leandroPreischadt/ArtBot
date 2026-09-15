from artbot.config.paths import MODELS_DIR
from artbot.config.settings import DEVICE, MODEL_NAME
from artbot.config.models.stt import whisper_model
from artbot.config.models.stt import *

MODEL_PATH = MODELS_DIR / MODEL_NAME


def main() -> None:
    print(f"MODEL: {MODEL_PATH}")
    print(f"DEVICE: {DEVICE}")
    
    loaded_model = load_model()
    
    while True:
        input("Press enter to start recording: ")
        my_text = whisper_model(loaded_model)


if __name__ == "__main__":
    main()
