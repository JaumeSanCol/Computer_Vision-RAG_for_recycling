# ingest.py
import bs4
import os
import shutil
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import DB_PATH, EMBEDDING_MODEL, URL_TARGET

def ingest_data():
    # Limpiar base de datos anterior si existe para evitar duplicados
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"Base de datos anterior eliminada en {DB_PATH}")

    print("1. Cargando documentos...")
    loader = WebBaseLoader(
        web_paths=(URL_TARGET,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
    )
    docs = loader.load()

    print("2. Dividiendo texto...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"   -> Creados {len(splits)} fragmentos.")

    print("3. Creando Embeddings y guardando en disco (esto puede tardar)...")
    # Al pasarle 'persist_directory', Chroma guarda los datos en disco automáticamente
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=DB_PATH
    )
    
    print(f"¡Listo! Base de datos vectorial guardada en '{DB_PATH}'")

if __name__ == "__main__":
    ingest_data()