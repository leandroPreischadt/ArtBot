from llama_cpp import Llama
from pathlib import Path
import json

from artbot.config.settings import MODEL_NAME, N_CTX

SYSTEM_PROMPT_PATH = Path(__file__).with_name("system_prompt.json")

def load_llm_model(): 
    MODEL_ID = MODEL_NAME
    
    llm = Llama(model_path=MODEL_ID,n_ctx=N_CTX,verbose=False )
    return llm

def llm_model(llm, text): 
    prompt_data = json.loads(SYSTEM_PROMPT_PATH.read_text(encoding="utf-8"))
    system_prompt = json.dumps(prompt_data, ensure_ascii=False, indent=2)

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
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
