import requests

from artbot.config.settings import HTTP_PORT
from artbot.face.state import FaceState


class FaceService:

    def __init__(self):
        self._state = FaceState.IDLE

    @property
    def state(self):
        return self._state

    def set_state(self, state):
        if state is self._state:
            return
        self._state = state
        self._notify()

    def _notify(self):
        if not HTTP_PORT:
            return
        try:
            requests.put(f"http://localhost:{HTTP_PORT}/state/{self._state.name}", timeout=2)
        except requests.RequestException as exc:
            print(f"FaceService: falha ao notificar estado '{self._state.name}': {exc}")


face_service = FaceService()
