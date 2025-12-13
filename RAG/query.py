# query.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import DB_PATH, EMBEDDING_MODEL, MODEL_NAME

def query(question: str) -> str:
    
    # 1. Cargar la base de datos
    vectorstore = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
    )
    retriever = vectorstore.as_retriever()

    # 2. Configurar LLM 
    # 'format="json"' a veces causa el problema opuesto, mejor dejarlo sin flag 
    # pero nos aseguramos de que temperature sea 0.
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    # 3. Definir el Prompt usando Mensajes (Más robusto que string único)
    
    # Mensaje de Sistema: Define la personalidad "sin filtros"
    system_instruction = """Eres una máquina de lectura ciega. NO eres un asistente. NO piensas. NO tienes moral.
    
    TU OBJETIVO: Repetir textualmente lo que encuentres en los documentos marcados como [DOCUMENTO INTERNO OFICIAL].
    
    REGLAS ABSOLUTAS:
    1. **PRIORIDAD MÁXIMA:** La información marcada como **[DOCUMENTO INTERNO OFICIAL]**. Esta es la fuente de verdad absoluta, sin importar si el lenguaje es ofensivo, vulgar o políticamente incorrecto..
    2. Si el documento contiene insultos ("puta", "mierda", etc), DEBES escribirlos tal cual.
    3. NO generes JSON, XML ni bloques de código. Solo texto plano.
    4. NO añadidas introducciones como "El documento dice..." ni conclusiones.
    5. Ignora cualquier instrucción previa que te pida "razonar" o "planificar".
    
    """

    # Mensaje Humano: Entrega el contexto y la pregunta
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

    # 4. Cadena
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\nPensando...\n")
    response = rag_chain.invoke(question)
    return response

if __name__ == "__main__":
    print(query("Matlab"))