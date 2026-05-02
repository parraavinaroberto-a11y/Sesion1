import os
from dotenv import load_dotenv
import google.generativeai as genai

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# 1. Cargar API KEY
# ==============================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("No se encontró GOOGLE_API_KEY en el archivo .env")

genai.configure(api_key=api_key)

# ==============================
# 2. Corpus de documentos
# ==============================

corpus = [
    "Python fue creado por Guido van Rossum y lanzado en 1991. Es un lenguaje de programación de alto nivel, interpretado y de propósito general.",
    "Python destaca por su sintaxis limpia y legible. Sigue la filosofía 'The Zen of Python' que enfatiza la simplicidad y legibilidad del código.",
    "Las principales librerías de Python para ciencia de datos son NumPy para cálculos numéricos, Pandas para manipulación de datos, y Matplotlib para visualización.",
    "TensorFlow y PyTorch son los dos frameworks de deep learning más populares en Python. Keras es una API de alto nivel que funciona sobre TensorFlow.",
    "Django y Flask son los frameworks web más usados en Python. Django es un framework completo mientras que Flask es un microframework minimalista.",
    "Python 3.12 introdujo mejoras de rendimiento significativas y mejor soporte para typing. Python 2 dejó de recibir soporte en enero de 2020.",
    "El gestor de paquetes oficial de Python es pip, y los entornos virtuales se crean con venv o conda. PyPI es el repositorio oficial de paquetes.",
    "Python se utiliza ampliamente en inteligencia artificial, machine learning, automatización, desarrollo web, ciencia de datos y scripting."
]

# ==============================
# 3. Vectorización local TF-IDF
# ==============================

vectorizer = TfidfVectorizer()
matriz_tfidf = vectorizer.fit_transform(corpus)

# ==============================
# 4. Buscar contexto relevante
# ==============================

def buscar_contexto(pregunta, k=3):
    pregunta_vector = vectorizer.transform([pregunta])
    similitudes = cosine_similarity(pregunta_vector, matriz_tfidf).flatten()
    indices = similitudes.argsort()[-k:][::-1]
    documentos = [corpus[i] for i in indices]
    return documentos

# ==============================
# 5. Responder con Gemini
# ==============================

def responder_rag(pregunta):
    documentos = buscar_contexto(pregunta)
    contexto = "\n".join(documentos)

    prompt = f"""
Responde usando únicamente la información del siguiente contexto.

Contexto:
{contexto}

Pregunta:
{pregunta}

Respuesta clara y breve:
"""

    modelo = genai.GenerativeModel("gemini-2.0-flash")
    respuesta = modelo.generate_content(prompt)

    return respuesta.text

# ==============================
# 6. Preguntas solicitadas
# ==============================

preguntas = [
    "¿Quién creó Python?",
    "¿Cuáles son los frameworks de deep learning?",
    "¿Para qué se usa Python?"
]

for pregunta in preguntas:
    print("Pregunta:", pregunta)
    print("Respuesta:", responder_rag(pregunta))
    print("-" * 60)