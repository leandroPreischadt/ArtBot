from llama_cpp import Llama, llama_supports_gpu_offload

from artbot.config.settings import (
    DEVICE,
    FLASH_ATTN,
    MAX_TOKENS,
    MODEL_NAME,
    N_BATCH,
    N_CTX,
    N_GPU_LAYERS,
    N_THREADS,
    N_THREADS_BATCH,
    N_UBATCH,
    OFFLOAD_KQV,
)

SYSTEM_PROMPT = (
    "Você é o Art, assistente virtual da Escola Politécnica da Univali. "
    "Responda sempre em português do Brasil, em um único parágrafo corrido. "
    "Não use listas, tópicos, markdown, negrito, títulos nem emojis. "
    "Responda com extensão proporcional à pergunta e cubra todos os itens pedidos. "
    "Quando o usuário pedir uma lista ou comparar vários itens, mencione cada item "
    "e todas as informações solicitadas, sem omitir itens por tentar ser breve. "
    "Use exclusivamente o contexto abaixo; se faltar informação, diga isso em uma frase. "
    "Não repita o contexto inteiro.\n\n"
    "Contexto:\n{context}"
)


def load_llm_model():
    if not MODEL_NAME:
        raise ValueError("MODEL_NAME não foi definido em environment/.env")

    if DEVICE == "cuda" and not llama_supports_gpu_offload():
        raise RuntimeError(
            "DEVICE=cuda, mas o llama-cpp-python foi instalado sem suporte CUDA. "
            "Reinstale-o com GGML_CUDA=1 antes de iniciar o ArtBot."
        )

    # ``None`` lets llama.cpp choose the host thread count.  Explicitly
    # passing n_gpu_layers=-1 is what places the complete GGUF on the Jetson.
    kwargs = {
        "model_path": MODEL_NAME,
        "n_ctx": N_CTX,
        "n_gpu_layers": N_GPU_LAYERS if DEVICE == "cuda" else 0,
        "n_batch": N_BATCH,
        "n_ubatch": N_UBATCH,
        "offload_kqv": OFFLOAD_KQV,
        "flash_attn": FLASH_ATTN,
        "verbose": False,
    }
    if N_THREADS > 0:
        kwargs["n_threads"] = N_THREADS
    if N_THREADS_BATCH > 0:
        kwargs["n_threads_batch"] = N_THREADS_BATCH

    print(
        "LLM backend: "
        f"{'CUDA' if DEVICE == 'cuda' else 'CPU'} | "
        f"gpu_layers={kwargs['n_gpu_layers']} | "
        f"ctx={N_CTX} | batch={N_BATCH}"
    )
    return Llama(**kwargs)


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
        max_tokens=MAX_TOKENS,
    )

    answer = response["choices"][0]["message"]["content"].strip()
    print(f"LLM: {answer}")
    return answer
