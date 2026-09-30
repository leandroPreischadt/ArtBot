from artbot.config.vector_database import getContext, populate_vector_database
from artbot.config.paths import MODELS_DIR
from artbot.config.settings import AUDIO_DEVICE, DEVICE, MODEL_NAME
from artbot.config.models.llm import *
from artbot.config.models.tts import load_piper_model, piper_model
from artbot.face.state import FaceState
from artbot.face_service import face_service

MODEL_PATH = MODELS_DIR / MODEL_NAME


def main() -> None:
    print(f"MODEL: {MODEL_PATH}")
    print(f"DEVICE: {DEVICE}")
    print(f"AUDIO_DEVICE: {AUDIO_DEVICE}")
    
    """Loading models"""
    loaded_llm_model = load_llm_model()
    loaded_tts_model = load_piper_model()
    populate_vector_database()


    """Executing programm"""
    while True:
        try:
            my_text = input("\nDigite sua pergunta (ou Ctrl+C para sair): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nArtBot encerrado.")
            break

        if not my_text:
            print("Digite uma pergunta para continuar.")
            continue

        face_service.set_state(FaceState.THINKING)

        context = getContext(my_text)

        llm_answer = llm_model(llm=loaded_llm_model, text=my_text, context=context)
        face_service.set_state(FaceState.SPEAKING)
        piper_model(voice=loaded_tts_model, text=llm_answer)

        face_service.set_state(FaceState.IDLE)
        
if __name__ == "__main__":
    main()
