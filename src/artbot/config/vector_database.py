import json

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
    busca = colecao.query(
        query_texts=[prompt],
        n_results=1,
        where={"categoria": {"$in": ["curso", "contexto", "ingresso"]}},
        include=["documents"],
    )["documents"][0] # type: ignore
    contexto = "\n".join(busca)
    print("Contexto utilizado: " + contexto)
    return contexto
