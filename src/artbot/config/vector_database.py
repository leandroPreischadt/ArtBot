import json
import re
import unicodedata

import chromadb

from artbot.config.settings import DATABASE_PATH, KNOWLEDGE_PATH


chroma_client = chromadb.PersistentClient(path=DATABASE_PATH)
colecao = chroma_client.get_or_create_collection(name="conhecimento_univali")


def populate_vector_database():
    knowledge = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    atuais = set(colecao.get()["ids"])
    novos = set(knowledge["ids"])
    if atuais == novos:
        return

    print("[STARTED] Populating vector")
    if atuais:
        colecao.delete(ids=list(atuais))
    colecao.add(
        documents=knowledge["documents"],
        ids=knowledge["ids"],
        metadatas=knowledge["metadatas"],
    )
    print("[FINISHED] Populating vector database")


def getContext(prompt):
    # Perguntas de catálogo precisam de todos os cursos. Uma busca semântica
    # comum retorna apenas o documento mais parecido (por exemplo, Construção
    # Naval), mesmo quando a pergunta pede a lista completa.
    pergunta = _normalizar(prompt)
    pede_catalogo = (
        "curso" in pergunta
        and any(
            termo in pergunta
            for termo in (
                "quais",
                "oferece",
                "ofert",
                "lista",
                "todos",
                "duracao deles",
                "duracao dos cursos",
            )
        )
    )

    if pede_catalogo:
        busca = colecao.get(
            where={"categoria": "curso"},
            include=["documents"],
        )["documents"]
    else:
        busca = colecao.query(
            query_texts=[prompt],
            n_results=6,
            where={"categoria": {"$in": ["curso", "contexto", "ingresso"]}},
            include=["documents"],
        )["documents"][0] # type: ignore

    contexto = "\n".join(busca)
    print("Contexto utilizado: " + contexto)
    return contexto


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )
    return re.sub(r"\s+", " ", texto).strip()
