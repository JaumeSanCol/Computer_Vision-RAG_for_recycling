# RAG/preload_database.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from preproces_data import extraer_por_titulos
from config import DB_PATH, EMBEDDING_MODEL, URL_TARGET, DATA_PATH

def preparar_base_datos():
    print(f"--- [CARGA] Procesando el PDF...")
    pdf_path = "RAG/docs/guia.pdf"
    
    # Extracción por títulos
    secciones_grandes = extraer_por_titulos(pdf_path)
    
    # Configuramos un splitter de refuerzo
    # El overlap de 100 hace que no se corten frases importantes
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    # Dividimos las secciones que se pasen de la raya
    docs_finales = text_splitter.split_documents(secciones_grandes)

    print(f"--- [PROCESAMIENTO] De {len(secciones_grandes)} secciones han salido {len(docs_finales)} fragmentos.")

    # Enviamos a la base de datos
    vectorstore = Chroma.from_documents(
        documents=docs_finales,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=DB_PATH
    )
    print("¡LISTO! Base de datos regenerada.")
if __name__ == "__main__":
    preparar_base_datos()