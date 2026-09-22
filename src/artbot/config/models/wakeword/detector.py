import queue

import numpy as np
import openwakeword
import sounddevice as sd
from openwakeword.model import Model

SAMPLE_RATE = 16000
CHANNELS = 1

FRAME_SAMPLES = 1280

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    audio_queue.put(indata.copy())

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

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=audio_callback,
        ):

            while True:

                frame = audio_queue.get()

                frame = frame.flatten()

                prediction = self.model.predict(frame)

                for wake_word, confidence in prediction.items():

                    if confidence >= 0.5:

                        print(
                            f"Wake word detectada: "
                            f"{wake_word} "
                            f"(confidence={confidence:.2f})"
                        )

                        return True