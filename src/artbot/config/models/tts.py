import html
import re

from piper import PiperVoice
import sounddevice as sd
from artbot.config.settings import TTS_MODEL_NAME, AUDIO_DEVICE

# Piper + processamento de áudio -> tentar fazer para uma voz mais robótica 

def load_piper_model():
    voice = PiperVoice.load(TTS_MODEL_NAME)
    return voice


def clean_text_for_speech(text: str) -> str:
    """Remove Markdown e transforma a resposta em texto natural para o Piper."""
    text = html.unescape(text)
    text = re.sub(r"```(?:\w+)?\s*|```", "", text)
    text = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        line = re.sub(r"[*_~`]+", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)

    # Uma quebra de linha do Markdown não garante pausa no TTS. Pontuamos
    # itens sem pontuação para que uma lista soe como frases separadas.
    for index, line in enumerate(lines):
        if line[-1] not in ".!?;:":
            lines[index] = f"{line}."

    return " ".join(lines).strip()


def piper_model(voice, text):
    text = clean_text_for_speech(text)
    with sd.RawOutputStream(
        device=AUDIO_DEVICE,
        samplerate=voice.config.sample_rate,
        channels=1,
        dtype="int16",
    ) as stream:
        for chunk in voice.synthesize(text):
            stream.write(chunk.audio_int16_bytes)
