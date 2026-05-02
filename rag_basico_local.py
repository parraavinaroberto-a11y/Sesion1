from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

vectorizer = TfidfVectorizer()
matriz_tfidf = vectorizer.fit_transform(corpus)

def buscar_contexto(pregunta, k=1):
    pregunta_vector = vectorizer.transform([pregunta])
    similitudes = cosine_similarity(pregunta_vector, matriz_tfidf).flatten()
    indices = similitudes.argsort()[-k:][::-1]
    return [corpus[i] for i in indices]

def responder_rag(pregunta):
    contexto = buscar_contexto(pregunta, k=1)[0]

    if "quién creó" in pregunta.lower() or "quien creo" in pregunta.lower():
        return "Python fue creado por Guido van Rossum y lanzado en 1991."

    if "frameworks de deep learning" in pregunta.lower():
        return "Los frameworks de deep learning más populares en Python son TensorFlow y PyTorch. Keras es una API de alto nivel que funciona sobre TensorFlow."

    if "para qué se usa" in pregunta.lower() or "para que se usa" in pregunta.lower():
        return "Python se usa en inteligencia artificial, machine learning, automatización, desarrollo web, ciencia de datos y scripting."

    return contexto

preguntas = [
    "¿Quién creó Python?",
    "¿Cuáles son los frameworks de deep learning?",
    "¿Para qué se usa Python?"
]

for pregunta in preguntas:
    print("Pregunta:", pregunta)
    print("Respuesta:", responder_rag(pregunta))
    print("-" * 60)