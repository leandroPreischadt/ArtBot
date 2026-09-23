from llama_cpp import Llama
from artbot.config.settings import MODEL_NAME, N_CTX

def load_llm_model(): 
    MODEL_ID = MODEL_NAME
    
    llm = Llama(model_path=MODEL_ID,n_ctx=N_CTX,verbose=False )
    return llm

def llm_model(llm, text): 
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "Seu nome é Art, uma abrevicão para ArtBot. Você é um assistente pessoal que responde as perguntas das pessoas de forma concisa e objetiva. Seja amigável e responda o mais correto possível não gere caracteres especiais a não ser aqueles que dão sentido para uma frase, ou seja, se você fizer uma pergunta você  precisa colocar um ponto de interrogação, ou exclamação, tudo depende do contexto, entretanto não gere emojis e caracteres que não tenham relação com sua resposta.",
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
