import logging
import time

from llama_cpp import Path

from artbot.config.vector_database import getContext, populate_vector_database
from artbot.config.paths import MODELS_DIR
from artbot.config.settings import AUDIO_DEVICE, DEVICE, MODEL_NAME
from artbot.config.models.stt import load_stt_model, whisper_model
from artbot.config.models.llm import *
from artbot.config.models.tts import load_piper_model, piper_model
from artbot.config.models.wakeword.detector import WakeWordDetector
from artbot.face.state import FaceState
from artbot.face_service import face_service

MODEL_PATH = MODELS_DIR / MODEL_NAME
logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        filemode="w",
        filename=Path(__file__).parent / "llm.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print(f"MODEL: {MODEL_PATH}")
    print(f"DEVICE: {DEVICE}")
    print(f"AUDIO_DEVICE: {AUDIO_DEVICE}")
    
    """Loading models"""
    loaded_stt_model = load_stt_model()
    loaded_llm_model = load_llm_model()
    loaded_tts_model = load_piper_model()
    wakeword = WakeWordDetector()
    populate_vector_database()

    ultima_interacao = None

    """Executing programm"""
    while True:
        try:
            wakeword.listen()
            print("Wake word detected")

            face_service.set_state(FaceState.THINKING)
            my_text = whisper_model(loaded_stt_model)

            if not my_text:
                continue

            context = getContext(my_text)
            previous_interaction = ultima_interacao
            ultima_interacao = None
            llm_answer = llm_model(
                llm=loaded_llm_model,
                text=my_text,
                context=context,
                previous_interaction=previous_interaction,
            )
            ultima_interacao = {
                "question": my_text,
                "answer": llm_answer,
            }
            face_service.set_state(FaceState.SPEAKING)
            piper_model(voice=loaded_tts_model, text=llm_answer)
        except KeyboardInterrupt:
            print("\nArtBot encerrado.")
            break
        except Exception:
            # Inclui falhas de áudio/STT/wake word, que antes estavam fora do
            # try e encerravam o subprocesso sem voltar ao modo de espera.
            logger.exception("Falha no ciclo de atendimento; tentando novamente")
            time.sleep(1)
        finally:
            try:
                face_service.set_state(FaceState.IDLE)
            except Exception:
                logger.exception("Não foi possível atualizar o estado da face")
        
if __name__ == "__main__":
    main()
