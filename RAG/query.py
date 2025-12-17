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
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
llm = ChatOllama(model=MODEL_NAME, temperature=0)

system_instruction = """Eres un experto en reciclaje y asistente documental estricto.
TU OBJETIVO: Indicar el método de reciclaje correcto basándote en el contexto o en la clasificación de residuos especiales.

INSTRUCCIONES DE PROCESAMIENTO:
1. MATERIALES EN CONTEXTO: Si el objeto es de papel, vidrio, plástico o alguno de los mencionados en la documentación, utiliza ÚNICAMENTE la información de los documentos para indicar el contenedor y las instrucciones (ej. vaciar líquido).
2. RESIDUOS ESPECIALES: Si el objeto NO aparece en los documentos pero es un residuo peligroso o voluminoso (pilas, baterías, aceite, madera, muebles, electrónicos), clasifícalo como "RESIDUO ESPECIAL".
3. PRIORIDAD DE ESTADO: Si el usuario menciona que el objeto está "lleno", "sucio" o "con restos de comida", prioriza la instrucción de limpieza antes del reciclaje.
4. MATERIALES MIXTOS PELIGROSOS: Si el objeto tiene componentes peligrosos (aceite, baterías) mezclados con materiales reciclables, prioriza tirarlo al contenedor de RESIDUO ESPECIAL.
5. MATERIALES MIXTOS GENERALES: Si el objeto tiene componentes biodegradables (comida) muy pegados o es "muy complejo de limpiar", prioriza tirarlo al contenedor de RESTOS/GRIS en lugar de intentar reciclar el material base.

REGLAS DE SALIDA:
- Contenedor: Indica el color (azul, verde, amarillo, gris) o "Punto Limpio" para residuos especiales.
- Acción: Si es necesario, añade una instrucción breve (ej. "Vaciar contenido").
- Respuesta Directa: Entrega solo la solución. Si no es un material del contexto ni un residuo especial claro, di: "La información no aparece en los documentos".
EJEMPLOS DE REFERENCIA:
- Usuario: "Botella de vidrio llena" -> Respuesta: Contenedor: Verde. Acción: Vaciar contenido antes de reciclar.
- Usuario: "Caja de pizza con mucha grasa" -> Respuesta: Contenedor: Gris (Restos). Acción: Tirar a restos por exceso de suciedad.
- Usuario: "Pilas gastadas" -> Respuesta: Contenedor: Punto Limpio. Acción: Residuo Especial.
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
    print(" -> Respuesta: ", end="\n", flush=True) # Preparamos la salida
    
    # USAMOS STREAM EN LUGAR DE INVOKE
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    query_stream("Botella de plástico sucia con restos de líquido")