"""Carrega a fonte única de instruções e conhecimento do ArtBot."""

import json
from functools import lru_cache
from typing import Any

from artbot.config.paths import BASE_DIR


PROMPT_PATH = BASE_DIR / "prompt.json"


@lru_cache(maxsize=1)
def load_prompt() -> dict[str, Any]:
    return json.loads(PROMPT_PATH.read_text(encoding="utf-8"))


def knowledge() -> dict[str, list[Any]]:
    prompt = load_prompt()
    return {
        "ids": prompt["ids"],
        "documents": prompt["documents"],
        "metadatas": prompt["metadatas"],
    }


def render_system_prompt(context: str) -> str:
    """Renderiza as regras fixas e o contexto relevante para cada pergunta."""
    prompt = load_prompt()["system_prompt"]
    regras = "\n".join(f"- {regra}" for regra in prompt["regras_de_resposta"])
    politicas = "\n".join(f"- {politica}" for politica in prompt["politicas"])

    return (
        f"{prompt['identidade']}\n\n"
        "Regras de resposta:\n"
        f"{regras}\n\n"
        "Políticas de escopo e precisão:\n"
        f"{politicas}\n\n"
        "Conhecimento relevante para esta pergunta:\n"
        f"{context}"
    )
