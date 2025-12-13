# query.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import DB_PATH, MODEL_NAME, EMBEDDING_MODEL

def query(question: str) -> str:
    
    # 1. Cargar la base de datos existente
    vectorstore = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL)
    )
    
    retriever = vectorstore.as_retriever()

    # 2. Configurar LLM y Prompt
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    template = """Responde a la pregunta basándote solo en el siguiente contexto:
    {context}

    Pregunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Crear cadena
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Generar respuesta
    print("\nPensando...\n")
    response = rag_chain.invoke(question)
    return response
