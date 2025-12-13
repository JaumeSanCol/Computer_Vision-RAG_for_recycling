# main.py

import argparse
from query import query


def main():
    # Permitir pasar la pregunta como argumento, o pedirla si no se da
    parser = argparse.ArgumentParser(description="Pregunta a tu RAG local")
    parser.add_argument("query", type=str, nargs="?", help="La pregunta que quieres hacer")
    args = parser.parse_args()

    # Si no hay argumento, pedimos input interactivo
    question = args.query
    if not question:
        print("\n--- Sistema RAG Local (Ollama) ---")
        question = input("Escribe tu pregunta: ")
    response= query(question)
    print("Respuesta:")
    print(response)

if __name__ == "__main__":
    main()