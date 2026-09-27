from artbot.config.paths import MODELS_DIR
from artbot.config.settings import DEVICE, MODEL_NAME
from artbot.config.models.stt import whisper_model
from artbot.config.models.stt import *
from artbot.config.models.llm import *
from artbot.config.models.tts import load_piper_model, piper_model
from artbot.config.models.wakeword.detector import *
from artbot.face.state import FaceState
from artbot.face_service import face_service

MODEL_PATH = MODELS_DIR / MODEL_NAME


def main() -> None:
    print(f"MODEL: {MODEL_PATH}")
    print(f"DEVICE: {DEVICE}")
    
    """Loading models"""
    loaded_stt_model = load_stt_model()
    loaded_llm_model = load_llm_model()
    loaded_tts_model = load_piper_model()
    wakeword = WakeWordDetector()
    
    """Executing programm"""
    while True:
        
        wakeword.listen()
        
        print("Wake word detected")
        
        face_service.set_state(FaceState.THINKING)
        my_text = whisper_model(loaded_stt_model)
        
        llm_answer = llm_model(llm=loaded_llm_model, text=my_text)
        
        face_service.set_state(FaceState.SPEAKING)
        piper_model(voice=loaded_tts_model, text=llm_answer)
        
        face_service.set_state(FaceState.IDLE)
        
if __name__ == "__main__":
    main()