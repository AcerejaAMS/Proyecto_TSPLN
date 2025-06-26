from transformers import MarianMTModel, MarianTokenizer

def cargar_modelo_traduccion(idioma_origen, idioma_destino):
    modelo_seleccionado = f'Helsinki-NLP/opus-mt-{idioma_origen}-{idioma_destino}'
    try:
        modelo = MarianMTModel.from_pretrained(modelo_seleccionado)
        tokenizer = MarianTokenizer.from_pretrained(modelo_seleccionado)
    except Exception as e:
        print(f"Error cargando modelo de traducción: {e}")
        return ' ', ' '
    return modelo, tokenizer

def traductor(texto, modelo, tokenizer):
    entrada = tokenizer(texto, return_tensors="pt", padding=True, truncation=True)
    traduccion = modelo.generate(
        **entrada,
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(traduccion[0], skip_special_tokens=True)
