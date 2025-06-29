import streamlit as st
from io import StringIO
import PyPDF2
import docx
import traductor
import chatbot

from chatbot import preparar_base_conocimiento, modelo_embed  # Importa función y modelo de embeddings

# Configuración de la página
st.set_page_config(page_title="Sistema de Traducción y Chatbot", layout="wide")
st.title("Chatbot de búsqueda y traducción multilingüe")

# Inicializar estado
if "historial_idioma_entrada" not in st.session_state:
    st.session_state.historial_idioma_entrada = []

if "historial_idioma_salida" not in st.session_state:
    st.session_state.historial_idioma_salida = []

if "documento" not in st.session_state:
    st.session_state.documento = ""

if "base_index" not in st.session_state:
    st.session_state.base_index = None

if "base_fragmentos" not in st.session_state:
    st.session_state.base_fragmentos = None

if "modelo_embed" not in st.session_state:
    st.session_state.modelo_embed = modelo_embed

# Selección de idiomas
idiomas = ['es', 'en', 'fr', 'de', 'it']
col1, col2 = st.columns(2)
with col1:
    idioma_entrada = st.selectbox("Idioma de pregunta", idiomas, index=0, key="entrada")

idiomas_salida = [i for i in idiomas if i != idioma_entrada]

if 'idioma_salida_prev' in st.session_state and st.session_state.idioma_salida_prev in idiomas_salida:
    salida_default = idiomas_salida.index(st.session_state.idioma_salida_prev)
else:
    salida_default = 0

with col2:
    idioma_salida = st.selectbox("Idioma de la base", idiomas_salida, index=salida_default, key="salida")

st.session_state.idioma_salida_prev = idioma_salida

# Subir documento
st.markdown("### Cargar documento para la base de conocimiento")
archivo = st.file_uploader("Selecciona un archivo (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])

if archivo:
    contenido = ""
    if archivo.type == "text/plain":
        stringio = StringIO(archivo.getvalue().decode("utf-8"))
        contenido = stringio.read()
    elif archivo.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(archivo)
        contenido = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    elif archivo.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(archivo)
        contenido = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    st.session_state.documento = contenido 
    st.success("Documento cargado correctamente.")
    st.text_area("Vista previa del contenido del documento", contenido, height=200)

    # Preparar base de conocimiento
    i1_i2_modelo, i1_i2_tokenizer = traductor.cargar_modelo_traduccion(idioma_entrada, idioma_salida)
    index, fragmentos = preparar_base_conocimiento(
        contenido, 
        traductor.traductor, 
        i1_i2_modelo, 
        i1_i2_tokenizer
    )
    st.session_state.base_index = index
    st.session_state.base_fragmentos = fragmentos
    if idioma_salida == 'es':
        carpeta = 'BasesConocimiento/Base_espanol.txt'
    elif idioma_salida == 'en':
        carpeta = 'BasesConocimiento/Base_espanol.txt'
    elif idioma_salida == 'fr':
        carpeta = 'BasesConocimiento/Base_espanol.txt'
    elif idioma_salida == 'de':
        carpeta = 'BasesConocimiento/Base_espanol.txt'
    elif idioma_salida == 'it':
        carpeta = 'BasesConocimiento/Base_espanol.txt'
    if carpeta:
        with open(carpeta) as f:
            contenido_referencia = f.read()
    if contenido_referencia:
        # Ojo: este ejemplo asume que todo el documento es un solo bloque
        traduccion_automatica = traductor.traductor(contenido, i1_i2_modelo, i1_i2_tokenizer)
        bleu_score = traductor.evaluar_traduccion(traduccion_automatica, contenido_referencia)
        print(f"**BLEU Score de la traducción automática contra la referencia:** {bleu_score:.2f}")


# Función principal que une traducción y respuesta
def union(idioma_entrada, idioma_salida, texto):
    i1_i2_modelo, i1_i2_tokenizer = traductor.cargar_modelo_traduccion(idioma_entrada, idioma_salida)
    i2_i1_modelo, i2_i1_tokenizer = traductor.cargar_modelo_traduccion(idioma_salida, idioma_entrada)

    base_index = st.session_state.get("base_index", None)
    base_textos = st.session_state.get("base_fragmentos", None)
    embedder = st.session_state.get("modelo_embed", None)

    respuestas = chatbot.chatbot(
        texto,
        traductor.traductor,
        i1_i2_tokenizer,
        i2_i1_tokenizer,
        i1_i2_modelo,
        i2_i1_modelo,
        base_conocimiento_index=base_index,
        base_conocimiento_textos=base_textos,
        modelo_embed=embedder
    )

    return respuestas  # prompt_traducido, respuesta_traducida, respuesta

# Función para enviar mensaje
def enviar_mensaje():
    texto = st.session_state.input_text
    if texto.strip() != "":
        respuestas = union(idioma_entrada, idioma_salida, texto)
        print(respuestas)

        # Historial en idioma de entrada
        st.session_state.historial_idioma_entrada.append(("Tú", texto))
        st.session_state.historial_idioma_entrada.append(("Chatbot", respuestas[2]))

        # Historial en idioma de salida
        st.session_state.historial_idioma_salida.append(("Tú", respuestas[0]))
        st.session_state.historial_idioma_salida.append(("Chatbot", respuestas[1]))

        st.session_state.input_text = ""

# Formulario para enviar mensaje
with st.form(key="form_chat", clear_on_submit=False):
    texto = st.text_input("Ingresa tu mensaje:", key="input_text")
    enviar = st.form_submit_button("Enviar", on_click=enviar_mensaje)

# Mostrar historiales
st.markdown("---")
st.subheader("Conversación")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"### Conversación en el idioma de la pregunta ({idioma_entrada})")
    for remitente, mensaje in st.session_state.historial_idioma_entrada:
        st.markdown(f"**{remitente}:** {mensaje}")

with col2:
    st.markdown(f"### Conversación en el idioma de base ({idioma_salida})")
    for remitente, mensaje in st.session_state.historial_idioma_salida:
        st.markdown(f"**{remitente}:** {mensaje}")