from llama_cpp import Llama
from artbot.config.settings import MODEL_NAME

def load_llm_model(): 
    MODEL_ID = MODEL_NAME
    
    llm = Llama(model_path=MODEL_ID,n_ctx=2048,verbose=False )
    return llm


def llm_model(llm, text): 
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "seja sempre conciso, responda o mais objetivamente possível. você está rodando localmente, em um script Python. não gere pontuação, você vai gerar uma frase que vai ser falada por um modelo de TTS",
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        max_tokens=4096,
    )

    answer = response["choices"][0]["message"]["content"].strip()
    print(f"LLM: {answer}")
    return answer
