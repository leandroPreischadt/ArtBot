from llama_cpp import Llama, llama_supports_gpu_offload

from artbot.config.prompt import render_system_prompt
from artbot.config.settings import (
    DEVICE,
    DETAIL_MAX_TOKENS,
    FLASH_ATTN,
    LIST_MAX_TOKENS,
    MAX_TOKENS,
    MODEL_NAME,
    N_BATCH,
    N_CTX,
    N_GPU_LAYERS,
    N_THREADS,
    N_THREADS_BATCH,
    N_UBATCH,
    OFFLOAD_KQV,
    SIMPLE_MAX_TOKENS,
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


def llm_model(llm, text, context, previous_interaction=None):
    if isinstance(context, list):
        context = "\n".join(context)

    max_tokens = _response_token_limit(text)
    messages = [
        {
            "role": "system",
            "content": render_system_prompt(context),
        },
    ]

    if previous_interaction:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": previous_interaction["question"],
                },
                {
                    "role": "assistant",
                    "content": previous_interaction["answer"],
                },
            ]
        )

    messages.append({"role": "user", "content": text})

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
    )

    answer = response["choices"][0]["message"]["content"].strip()
    print(f"LLM: {answer}")
    return answer


def _response_token_limit(text: str) -> int:
    """Escolhe o limite conforme a amplitude pedida na pergunta."""
    normalized = text.casefold()
    list_request = any(
        termo in normalized
        for termo in ("quais", "lista", "todos", "todas", "cada", "formas de ingresso")
    )
    comparison_request = any(
        termo in normalized
        for termo in ("compare", "comparar", "comparação", "diferença", "diferenças")
    )
    detail_request = any(
        termo in normalized
        for termo in ("explique", "explica", "detalhe", "detalhes", "como funciona", "por que")
    )

    if comparison_request:
        return min(MAX_TOKENS, max(LIST_MAX_TOKENS, DETAIL_MAX_TOKENS))
    if list_request:
        return min(MAX_TOKENS, LIST_MAX_TOKENS)
    if detail_request:
        return min(MAX_TOKENS, DETAIL_MAX_TOKENS)
    return min(MAX_TOKENS, SIMPLE_MAX_TOKENS)
