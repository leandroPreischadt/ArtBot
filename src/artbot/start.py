"""Sobe a face e o serviço principal do ArtBot juntos.

Uso: uv run artbot-start
Ctrl+C (ou fechar qualquer um dos dois) encerra ambos.
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
    face = subprocess.Popen([sys.executable, "-m", "artbot.face.main"])

    if wait_for_face():
        print("Face pronta. Iniciando o ArtBot...")
    else:
        print("Face nao respondeu a tempo. Iniciando o ArtBot mesmo assim...")

    voice = subprocess.Popen([sys.executable, "-m", "artbot.main"])

    try:
        while face.poll() is None and voice.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in (face, voice):
            if process.poll() is None:
                process.terminate()
        print("ArtBot encerrado.")


if __name__ == "__main__":
    main()
