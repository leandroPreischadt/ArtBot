import re
import unicodedata

import chromadb

from artbot.config.prompt import knowledge
from artbot.config.settings import DATABASE_PATH


chroma_client = chromadb.PersistentClient(path=DATABASE_PATH)
colecao = chroma_client.get_or_create_collection(name="conhecimento_univali")


def populate_vector_database():
    dados = knowledge()
    atuais = colecao.get(include=["documents", "metadatas"])
    novos = {
        "ids": dados["ids"],
        "documents": dados["documents"],
        "metadatas": dados["metadatas"],
    }
    estado_atual = {
        "ids": atuais["ids"],
        "documents": atuais["documents"],
        "metadatas": atuais["metadatas"],
    }
    if estado_atual == novos:
        return

    print("[STARTED] Populating vector")
    if atuais["ids"]:
        colecao.delete(ids=atuais["ids"])
    colecao.add(
        documents=dados["documents"],
        ids=dados["ids"],
        metadatas=dados["metadatas"],
    )
    print("[FINISHED] Populating vector database")


def getContext(prompt):
    pergunta = _normalizar(prompt)
    dados_cursos = colecao.get(
        where={"categoria": "curso"},
        include=["documents", "metadatas"],
    )
    cursos = list(zip(dados_cursos["documents"], dados_cursos["metadatas"]))

    # Uma busca semântica é boa para uma dúvida aberta, mas não para um
    # catálogo: ela pode retornar só o curso mais parecido. Para catálogo,
    # enviamos ao modelo uma ficha curta de todos os cursos.
    pede_catalogo = _pede_catalogo(pergunta)

    if pede_catalogo:
        return _catalogo_resumido(cursos)

    # Quando o nome aparece na pergunta, não há motivo para pedir ao Chroma
    # seis documentos parecidos. O texto completo de um único curso também
    # dá uma resposta melhor para perguntas como "o que é e quanto dura?".
    curso = _curso_mencionado(pergunta, cursos)
    if curso is not None:
        return curso

    busca = colecao.query(
        query_texts=[prompt],
        n_results=3,
        where={"categoria": {"$in": ["curso", "contexto", "ingresso"]}},
        include=["documents"],
    )["documents"][0]  # type: ignore

    return "\n".join(busca)


def _pede_catalogo(pergunta: str) -> bool:
    termos_de_lista = (
        "quais",
        "oferece",
        "ofert",
        "lista",
        "todos",
        "todas",
        "cada",
        "duracao deles",
        "duracao dos cursos",
    )
    assunto_cursos = (
        "curso",
        "cursos",
        "politecnica",
        "graduacao",
        "mensalidade",
        "mensalidades",
        "custo",
        "custos",
        "duracao",
    )
    return any(termo in pergunta for termo in termos_de_lista) and any(
        termo in pergunta for termo in assunto_cursos
    )


def _curso_mencionado(pergunta: str, cursos: list[tuple[str, dict]]) -> str | None:
    # Nomes maiores primeiro evita que uma parte de um nome capture a pergunta
    # antes do nome completo, por exemplo, "engenharia de computação".
    cursos_ordenados = sorted(
        cursos,
        key=lambda curso: len(str(curso[1].get("nome", ""))),
        reverse=True,
    )
    for documento, metadata in cursos_ordenados:
        nome = metadata.get("nome", "")
        if nome and _normalizar(nome) in pergunta:
            return documento

    aliases = {
        "ads": "analise e desenvolvimento de sistemas",
        "analise de sistemas": "analise e desenvolvimento de sistemas",
        "ia": "inteligencia artificial",
    }
    for alias, nome_curso in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", pergunta):
            for documento, metadata in cursos:
                if _normalizar(metadata.get("nome", "")) == nome_curso:
                    return documento
    return None


def _catalogo_resumido(cursos: list[tuple[str, dict]]) -> str:
    linhas = [
        "Catálogo da Escola Politécnica da Univali. Liste todos os cursos abaixo e informe a duração.",
    ]
    for modalidade in ("Presencial", "EAD"):
        linhas.append(f"{modalidade}:")
        for documento, metadata in cursos:
            if metadata.get("modalidade") != modalidade:
                continue
            nome = metadata.get("nome", "Curso")
            duracao = re.search(r"Duração: ([^.]+)", documento)
            duracao_texto = duracao.group(1) if duracao else "não informada"
            linhas.append(f"{nome} | {duracao_texto}")
    linhas.append(
        "A base não informa valores de mensalidade ou outros custos. Não invente preços; diga isso brevemente se a pergunta pedir custos."
    )
    return "\n".join(linhas)


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )
    return re.sub(r"\s+", " ", texto).strip()
