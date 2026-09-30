from llama_cpp import Llama

from artbot.config.settings import MODEL_NAME, N_CTX

SYSTEM_PROMPT = (
    "Você é o Art, assistente virtual da Escola Politécnica da Univali. "
    "Responda sempre em português do Brasil, em um único parágrafo corrido. "
    "Não use listas, tópicos, markdown, negrito, títulos nem emojis. "
    "Seja breve: só o essencial para responder a pergunta. "
    "Use exclusivamente o contexto abaixo; se faltar informação, diga isso em uma frase. "
    "Não repita o contexto inteiro.\n\n"
    "Contexto:\n{context}"
)


def load_llm_model():
    return Llama(model_path=MODEL_NAME, n_ctx=N_CTX, verbose=False)


def llm_model(llm, text, context):
    if isinstance(context, list):
        context = "\n".join(context)

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(context=context),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        max_tokens=180,
    )

    answer = response["choices"][0]["message"]["content"].strip()
    print(f"LLM: {answer}")
    return answer
