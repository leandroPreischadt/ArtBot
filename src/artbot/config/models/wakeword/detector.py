import queue
import numpy as np
import openwakeword
import sounddevice as sd
from openwakeword.model import Model

from artbot.config.settings import AUDIO_DEVICE, WAKEWORD_THRESHOLD, NUMBER_OF_CHANNELS

# Variáveis do Hardware (DJI Mic)
SAMPLE_RATE_HW = 48000 
CHANNELS_HW = NUMBER_OF_CHANNELS
FRAME_SAMPLES_HW = 3840 # Equivalente a 1280 (exigido pelo OpenWakeWord) * 3

audio_queue = queue.Queue()


def _resolve_input_device():
    """Resolve o microfone padrão sem depender do alias ``default``."""
    if AUDIO_DEVICE is not None:
        return AUDIO_DEVICE

    default_input = sd.default.device[0]
    if isinstance(default_input, int) and default_input >= 0:
        return default_input

    for index, device in enumerate(sd.query_devices()):
        if int(device["max_input_channels"]) > 0:
            return index

    raise RuntimeError(
        "Nenhum dispositivo de entrada foi encontrado. "
        "Verifique a permissão de microfone do Terminal no macOS."
    )

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    # O DJI entrega estéreo em 48 kHz; o OpenWakeWord recebe mono em 16 kHz.
    mono = indata.mean(axis=1).astype(np.int16)
    audio_queue.put(mono[::3].copy())

class WakeWordDetector:

    def __init__(self):
        self.model = Model()

    def _clear_queue(self):
            while True:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break

    def _reset_model_state(self):
        self.model.reset()

        preprocessor = self.model.preprocessor

        preprocessor.raw_data_buffer.clear()
        preprocessor.melspectrogram_buffer = np.ones((76, 32))
        preprocessor.accumulated_samples = 0
        preprocessor.feature_buffer = preprocessor._get_embeddings(
            np.zeros(160000, dtype=np.int16)
        )

    def listen(self):
        self._clear_queue()
        self._reset_model_state()

        print("Waiting wake word...")

        input_device = _resolve_input_device()
        max_input_channels = int(
            sd.query_devices(input_device)["max_input_channels"]
        )
        input_channels = min(CHANNELS_HW, max_input_channels)

        with sd.InputStream(
            device=input_device,
            samplerate=SAMPLE_RATE_HW,
            channels=input_channels,
            dtype="int16",
            blocksize=FRAME_SAMPLES_HW, # Passando a variável de hardware correta
            callback=audio_callback,
        ):

            while True:
                frame = audio_queue.get()

                # O frame já sai do callback reduzido, pronto para a IA
                frame = frame.flatten()

                prediction = self.model.predict(frame)

                for wake_word, confidence in prediction.items():
                    if confidence >= WAKEWORD_THRESHOLD:
                        print(
                            f"Wake word detectada: "
                            f"{wake_word} "
                            f"(confidence={confidence:.2f})"
                        )
                        return True
