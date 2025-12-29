# config.py

# Carpeta donde se guardará la base de datos vectorial
DB_PATH = "RAG/chroma_db"

# Modelos a usar en Ollama
#MODEL_NAME = "llama3"             
MODEL_NAME = "phi3:mini"
EMBEDDING_MODEL = "nomic-embed-text" 

# Fuente de datos
URL_TARGET = "https://lilianweng.github.io/posts/2023-06-23-agent/"
DATA_PATH = "RAG/docs"  # Carpeta local con PDFs para cargar