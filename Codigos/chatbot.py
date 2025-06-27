from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Modelo de chatbot basado en DialoGPT
dialogo_tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-medium")
dialogo_modelo = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-medium")

# Modelo de embeddings
modelo_embed = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Función para leer base de conocimiento y traducir fragmentos
def preparar_base_conocimiento(texto_largo, traductor_func, modelo_trad, tokenizer_trad):
    fragmentos = [p.strip() for p in texto_largo.split('\n') if p.strip()]
    fragmentos_traducidos = [traductor_func(p, modelo_trad, tokenizer_trad) for p in fragmentos]
    vectores = modelo_embed.encode(fragmentos_traducidos)
    index = faiss.IndexFlatL2(vectores.shape[1])
    index.add(np.array(vectores))
    return index, fragmentos_traducidos


def generar_respuesta(prompt):
    entrada = dialogo_tokenizer.encode(prompt + " ", return_tensors='pt')
    salida = dialogo_modelo.generate(entrada, max_length=1000, pad_token_id=dialogo_tokenizer.eos_token_id)
    
    return dialogo_tokenizer.decode(salida[0], skip_special_tokens=True)

def chatbot(prompt, traductor_func, i1_i2_tokenizer, i2_i1_tokenizer, i1_i2_modelo, i2_i1_modelo,
            base_conocimiento_index=None, base_conocimiento_textos=None, modelo_embed=None):
    try:
        # Traducir pregunta al idioma objetivo
        prompt_traducido = traductor_func(prompt, i1_i2_modelo, i1_i2_tokenizer)

        # Buscar contexto si hay base de conocimiento
        contexto = ""
        if base_conocimiento_index is not None and base_conocimiento_textos is not None and modelo_embed is not None:
            emb = modelo_embed.encode([prompt_traducido])
            D, I = base_conocimiento_index.search(np.array(emb), k=3)
            contexto = "\n".join([base_conocimiento_textos[i] for i in I[0]])

        # Crear prompt combinado con contexto
        if contexto:
            prompt_con_contexto = f"Contexto:\n{contexto}\n\nPregunta:\n{prompt_traducido}"
        else:
            prompt_con_contexto = prompt_traducido

        # Generar respuesta en idioma de salida
        respuesta_traducida = generar_respuesta(prompt_con_contexto)

        # Traducir de nuevo al idioma original
        respuesta = traductor_func(respuesta_traducida, i2_i1_modelo, i2_i1_tokenizer)

        return [prompt_traducido, respuesta_traducida, respuesta]
    except Exception as e:
        return [f'Error: {str(e)}']
