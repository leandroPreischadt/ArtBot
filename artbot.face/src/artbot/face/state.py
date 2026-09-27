from enum import Enum, auto


class FaceState(Enum):
    IDLE = auto()
    SPEAKING = auto()
    THINKING = auto()
    BORED = auto()
