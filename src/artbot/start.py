"""Sobe a face e o serviço principal do ArtBot juntos.

Uso: uv run artbot-start
Ctrl+C encerra ambos. Se a janela da face falhar, o serviço de voz continua.
"""

import subprocess
import sys
import time

import requests

from artbot.config.settings import HTTP_PORT

FACE_STATE_URL = f"http://127.0.0.1:{HTTP_PORT}/state"


def wait_for_face(timeout: float = 15) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            requests.get(FACE_STATE_URL, timeout=1)
            return True
        except requests.RequestException:
            time.sleep(0.5)
    return False


def main() -> None:
    print("Iniciando a face...")
    # -u faz com que o traceback do subprocesso apareça imediatamente no
    # terminal, em vez de ficar preso no buffer até o processo morrer.
    face = subprocess.Popen([sys.executable, "-u", "-m", "artbot.face.main"])

    if wait_for_face():
        print("Face pronta. Iniciando o ArtBot...")
    else:
        print("Face nao respondeu a tempo. Iniciando o ArtBot mesmo assim...")

    voice = subprocess.Popen([sys.executable, "-u", "-m", "artbot.main"])

    try:
        # A face é uma interface auxiliar. Uma falha do Arcade não deve
        # derrubar o loop de voz, que consegue continuar sem animação.
        face_failure_reported = False
        while voice.poll() is None:
            if face.poll() is not None and not face_failure_reported:
                print(
                    "Aviso: o processo da face terminou; "
                    "o ArtBot continuará sem animação."
                )
                face_failure_reported = True
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        face_code = face.poll()
        voice_code = voice.poll()
        if face_code is not None:
            print(f"Processo da face terminou com código {face_code}.")
        if voice_code is not None:
            print(f"Processo de voz terminou com código {voice_code}.")

        for process in (face, voice):
            if process.poll() is None:
                process.terminate()
        print("ArtBot encerrado.")


if __name__ == "__main__":
    main()
