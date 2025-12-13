# config.py
import os

# Carpeta donde se guardará la base de datos vectorial
DB_PATH = "RAG/chroma_db"

# Modelos a usar en Ollama
MODEL_NAME = "llama3"              # Para generar texto
EMBEDDING_MODEL = "nomic-embed-text" # Para vectores (o "llama3" si no bajaste nomic)

# URL de ejemplo (puedes cambiar esto luego por PDFs)
URL_TARGET = "https://lilianweng.github.io/posts/2023-06-23-agent/"