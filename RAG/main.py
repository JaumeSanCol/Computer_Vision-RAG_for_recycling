# main.py

import argparse
from query import query_stream
import time

def main():
    # Permitir pasar la pregunta como argumento, o pedirla si no se da
    parser = argparse.ArgumentParser(description="Pregunta a tu RAG local")
    parser.add_argument("query", type=str, nargs="?", help="La pregunta que quieres hacer")
    args = parser.parse_args()
    # Si no hay argumento, pedimos input interactivo
    question = args.query
    while question != "exit":
        if not question:
            print("\n--- Sistema RAG Local ---")
        question = input("Escribe tu pregunta: ")
        if question == "exit":
            print("Saliendo...")
            break
        start=time.time()   
        query_stream(question)
        end=time.time()
        print(f"(Tiempo de respuesta: {end-start:.2f} segundos)")


if __name__ == "__main__":
    main()