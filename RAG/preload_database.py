import os
import shutil
import bs4

# Loaders
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, PyPDFLoader, TextLoader
# Core y Vector Store
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configuración
from config import DB_PATH, EMBEDDING_MODEL, URL_TARGET, DATA_PATH

def get_web_content():
    """
    Función general: 
    Extrae información básica de la web definida.
    Retorna: Lista de documentos brutos.
    """
    print(f"--- [WEB] Iniciando carga desde: {URL_TARGET}")
    try:
        loader = WebBaseLoader(
            web_paths=(URL_TARGET,),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            ))
        )
        docs = loader.load()
        for doc in docs:
            # 1. Metadatos para filtrado posterior
            doc.metadata["category"] = "general"
            doc.metadata["priority"] = "low"
            
            # 2. Modificación del contenido (Opcional pero recomendado)
            # Añadimos una cabecera para que el LLM sepa que esto es info pública
            doc.page_content = f"[FUENTE PÚBLICA/WEB] {doc.page_content}"

        print(f"   -> [WEB] {len(docs)} documentos marcados como 'general'.")
        return docs
    except Exception as e:
        print(f"   -> [WEB] Error cargando la web: {e}")
        return []
def get_local_content():
    """
    Función especializada:
    Extrae información específica de documentos PDF y TXT locales.
    """
    print(f"--- [LOCAL] Buscando documentos en: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"   -> [LOCAL] Alerta: La carpeta {DATA_PATH} no existe.")
        return []

    local_docs = []

    # -----------------------------------------------------------
    # 1. Cargar PDFs
    # -----------------------------------------------------------
    try:
        loader_pdf = DirectoryLoader(
            DATA_PATH, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        docs_pdf = loader_pdf.load()
        print(f"   -> Encontrados {len(docs_pdf)} PDFs.")
        local_docs.extend(docs_pdf)
    except Exception as e:
        print(f"   -> Error leyendo PDFs: {e}")

    # -----------------------------------------------------------
    # 2. Cargar TXT
    # -----------------------------------------------------------
    try:
        # loader_kwargs={'autodetect_encoding': True} ayuda si tienes archivos con ñ o tildes
        loader_txt = DirectoryLoader(
            DATA_PATH, 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            loader_kwargs={'autodetect_encoding': True}, 
            show_progress=True
        )
        docs_txt = loader_txt.load()
        print(f"   -> Encontrados {len(docs_txt)} TXTs.")
        local_docs.extend(docs_txt)
    except Exception as e:
        print(f"   -> Error leyendo TXTs: {e}")

    # -----------------------------------------------------------
    # 3. Procesamiento y Etiquetado (Común para todos)
    # -----------------------------------------------------------
    if not local_docs:
        print("   -> [LOCAL] No se encontraron archivos válidos.")
        return []

    for doc in local_docs:
        # A. Metadatos de prioridad
        doc.metadata["category"] = "specialized"
        doc.metadata["priority"] = "high"
        
        # B. Inyección de autoridad en el texto
        # Esto aplicará tanto a los PDFs como a los TXTs
        doc.page_content = f"[DOCUMENTO INTERNO OFICIAL] {doc.page_content}"
        
    print(f"   -> [LOCAL] Total: {len(local_docs)} documentos marcados como 'specialized'.")
    return local_docs

def create_vector_db(all_docs):
    """
    Función Orquestadora:
    Recibe todos los documentos, limpia la BD antigua y genera la nueva.
    """
    if not all_docs:
        print("!!! No hay documentos para procesar. Abortando.")
        return

    # Limpieza de BD existente
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"--- [DB] Base de datos anterior eliminada en {DB_PATH}")

    # Splitting (Dividir texto)
    print("--- [PROCESAMIENTO] Dividiendo textos...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"   -> Total de fragmentos creados: {len(splits)}")

    # Guardado en Chroma
    print("--- [PROCESAMIENTO] Generando Embeddings (esto tarda)...")
    Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=DB_PATH
    )
    print(f"¡LISTO! Base de datos regenerada en '{DB_PATH}'")

if __name__ == "__main__":
    # --- EJECUCIÓN PRINCIPAL ---
    
    # Obtenemos datos de la fuente general
    docs_web = get_web_content()
    
    # Obtenemos datos de la fuente específica
    docs_local = get_local_content()
    
    # Juntamos todo en una sola lista
    full_corpus = docs_web + docs_local
    
    # Creamos la base de datos
    create_vector_db(full_corpus)