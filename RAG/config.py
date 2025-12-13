# config.py
import os

# Carpeta donde se guardará la base de datos vectorial
DB_PATH = "RAG/chroma_db"

# Modelos a usar en Ollama
#MODEL_NAME = "llama3"              # Para generar texto
MODEL_NAME = "dolphin-mistral"
EMBEDDING_MODEL = "nomic-embed-text" # Para vectores (o "llama3" si no bajaste nomic)

# Fuente de datos
URL_TARGET = "https://lilianweng.github.io/posts/2023-06-23-agent/"
DATA_PATH = "RAG/docs"  # Carpeta local con PDFs para cargar