# query.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import DB_PATH, EMBEDDING_MODEL, MODEL_NAME

# ... (imports anteriores)
import sys

# MUEVE ESTO FUERA DE LA FUNCIÓN (Ver Solución 2)
vectorstore = Chroma(
    persist_directory=DB_PATH, 
    embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatOllama(model=MODEL_NAME, temperature=0)

system_instruction = """Eres un asistente documental estricto.
TU OBJETIVO: Responder a la pregunta del usuario USANDO ÚNICAMENTE el contexto proporcionado.

INSTRUCCIONES DE PROCESAMIENTO INTERNO (NO IMPRIMIR):
1. Analiza la pregunta y localiza la evidencia en el contexto.
2. Filtra la información priorizando [DOCUMENTO INTERNO OFICIAL].
3. Verifica que no estás inventando nada ni citando fuentes externas prohibidas.
4. Redacta la respuesta final.

REGLAS DE SALIDA OBLIGATORIAS:
1. **Respuesta Directa:** Entrega ÚNICAMENTE el dato o la frase que responde a la pregunta.
2. **Cero Justificaciones:** ESTÁ PROHIBIDO explicar por qué respondes de esa manera. No digas "Basado en las reglas...", "Para no mencionar la empresa...", ni "El contexto indica...".
3. **Sin Alucinaciones:** Si la respuesta no está, escribe EXACTAMENTE: "La información no aparece en los documentos".
4. **Limpieza:** Extrae la respuesta lo más literal posible sin copiar bloques innecesarios.
"""

human_instruction = """Contexto:
{context}

Pregunta: {question}
Respuesta literal:"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_instruction),
    HumanMessagePromptTemplate.from_template(human_instruction)
])

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


# Cadena optimizada
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def query_stream(question: str):
    print(" -> Respuesta: ", end="", flush=True) # Preparamos la salida
    
    # USAMOS STREAM EN LUGAR DE INVOKE
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    query_stream("Matlab")