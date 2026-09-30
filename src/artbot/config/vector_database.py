import json
from pathlib import Path

import chromadb

from artbot.config.settings import DATABASE_PATH, KNOWLEDGE_PATH


chroma_client = chromadb.PersistentClient(path=DATABASE_PATH)
colecao = chroma_client.get_or_create_collection(name="conhecimento_univali")


def populate_vector_database():
    if colecao.count() > 0:
        return

    print("[STARTED] Populating vector")
    knowledge = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    colecao.add(
        documents=knowledge["documents"],
        ids=knowledge["ids"],
        metadatas=knowledge["metadatas"],
    )
    print("[FINISHED] Populating vector database")


def getContext(prompt):
    busca = colecao.query(query_texts=[prompt], include=["documents"])["documents"][0][0]  # type: ignore

    print("Contexto utilizado: " + busca)
    return busca
