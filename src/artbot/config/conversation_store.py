"""Registro das conversas do ArtBot em um arquivo de texto."""

from datetime import datetime, timezone
from pathlib import Path


class ConversationStore:
    """Salva as falas do usuário e da LLM com horários distintos."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append(self, text: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(text)

    def save_question(self, question: str) -> str:
        """Salva a fala do usuário quando a transcrição termina."""
        interaction_id = self._now()
        self._append(
            f"[{interaction_id}] USUÁRIO | perguntou em {interaction_id}: {question}\n"
        )
        return interaction_id

    def save_answer(self, interaction_id: str, answer: str) -> None:
        """Salva a fala da LLM quando a geração termina."""
        answered_at = self._now()
        self._append(
            f"[{interaction_id}] LLM | respondeu em {answered_at}: {answer}\n\n"
        )
